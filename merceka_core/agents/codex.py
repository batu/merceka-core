from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merceka_core import _cli
from merceka_core.agent import (
  AgentComplete,
  AgentProfile,
  AgentRawProviderEvent,
  AgentRequest,
  AgentResult,
  AgentStreamEvent,
  AgentTextDelta,
  ProviderFailure,
  RawProviderEvent,
)

CODEX_PROVIDER = "codex"
CODEX_TIMEOUT_SECONDS = 300
DEFAULT_CODEX_MODEL_ALIASES = {"", "default", "codex-default", "codex-default-high", "gpt-5.5-high"}


@dataclass(frozen=True)
class CodexAgentProvider:
  model: str = "codex-default-high"
  codex_binary: str = "codex"
  timeout_seconds: int = CODEX_TIMEOUT_SECONDS

  async def run(self, request: AgentRequest) -> AgentResult:
    return await asyncio.to_thread(self._run_sync, request)

  def stream(self, request: AgentRequest) -> AsyncIterator[AgentStreamEvent]:
    return self._stream(request)

  def _run_sync(self, request: AgentRequest) -> AgentResult:
    with tempfile.NamedTemporaryFile("r", encoding="utf-8", delete=False) as output_file:
      output_path = Path(output_file.name)
    try:
      cmd = self._command(request, json_output=True)
      cmd.extend(["--output-last-message", str(output_path)])
      result = subprocess.run(
        cmd,
        input=self._prompt(request),
        capture_output=True,
        text=True,
        timeout=self.timeout_seconds,
        cwd=str(request.roots[0]),
      )
      raw_events = tuple(self._raw_events_from_stdout(result.stdout))
      if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown provider error"
        raise ProviderFailure(f"Codex failed with exit {result.returncode}: {message}")
      text = output_path.read_text(encoding="utf-8").strip()
      if not raw_events:
        raw_events = (
          RawProviderEvent(
            provider=CODEX_PROVIDER,
            event_type="result",
            payload={"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode},
          ),
        )
      return AgentResult(text=text, raw_events=raw_events)
    finally:
      output_path.unlink(missing_ok=True)

  async def _stream(self, request: AgentRequest) -> AsyncIterator[AgentStreamEvent]:
    cmd = self._command(request, json_output=True)
    process = subprocess.Popen(
      cmd,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      bufsize=1,
      cwd=str(request.roots[0]),
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
      raise ProviderFailure("Codex stream did not expose stdio pipes")

    process.stdin.write(self._prompt(request))
    process.stdin.close()

    raw_events: list[RawProviderEvent] = []
    text_chunks: list[str] = []
    try:
      while True:
        line = await asyncio.to_thread(process.stdout.readline)
        if line == "":
          break
        line = line.strip()
        if not line:
          continue

        raw_event = self._raw_event_from_line(line)
        raw_events.append(raw_event)
        yield AgentRawProviderEvent(raw_event=raw_event)

        payload = raw_event.payload
        if isinstance(payload, dict):
          text = self._text_delta_from_payload(payload)
          if text is not None:
            text_chunks.append(text)
            yield AgentTextDelta(content=text)

      returncode = await asyncio.to_thread(process.wait)
      stderr = await asyncio.to_thread(process.stderr.read)
      if returncode != 0:
        message = stderr.strip() or f"exit {returncode}"
        raise ProviderFailure(f"Codex stream failed with exit {returncode}: {message}")
      yield AgentComplete(result=AgentResult(text="".join(text_chunks), raw_events=tuple(raw_events)))
    except GeneratorExit:
      self._terminate_process(process)
      raise
    except asyncio.CancelledError:
      self._terminate_process(process)
      raise
    finally:
      if process.returncode is None:
        self._terminate_process(process)
      self._close_pipe(process.stdout)
      self._close_pipe(process.stderr)

  def _command(self, request: AgentRequest, *, json_output: bool) -> list[str]:
    model = "" if self.model in DEFAULT_CODEX_MODEL_ALIASES else self.model
    return _cli.codex_exec_command(
      model,
      sandbox="workspace-write" if request.profile == AgentProfile.WRITE else "read-only",
      cd=str(request.roots[0]),
      add_dirs=[str(root) for root in request.roots[1:]],
      json_output=json_output,
      reasoning_effort="high",
      binary=self.codex_binary,
    )

  def _prompt(self, request: AgentRequest) -> str:
    if request.profile == AgentProfile.WRITE:
      guidance = (
        "You are running under a write profile. Read/search and modify files only within "
        "declared roots.\n\n"
      )
    else:
      guidance = (
        "You are running under a read-only profile. Read/search only within declared roots. "
        "Do not modify files.\n\n"
      )
    return (
      f"<system>\n{request.system_prompt}\n</system>\n\n"
      f"{guidance}"
      f"<user>\n{request.message}\n</user>\n"
    )

  def _raw_events_from_stdout(self, stdout: str) -> list[RawProviderEvent]:
    return [self._raw_event_from_line(line) for line in stdout.splitlines() if line.strip()]

  def _raw_event_from_line(self, line: str) -> RawProviderEvent:
    try:
      payload: Any = json.loads(line)
    except json.JSONDecodeError as exc:
      return RawProviderEvent(
        provider=CODEX_PROVIDER,
        event_type="malformed_json",
        payload={"line": line, "error": str(exc)},
      )
    event_type = str(payload.get("type", "raw")) if isinstance(payload, dict) else "raw"
    return RawProviderEvent(provider=CODEX_PROVIDER, event_type=event_type, payload=payload)

  def _text_delta_from_payload(self, payload: dict[str, Any]) -> str | None:
    for key in ("delta", "text", "message", "content"):
      value = payload.get(key)
      if isinstance(value, str) and value:
        return value
    message = payload.get("message")
    if isinstance(message, dict):
      content = message.get("content")
      if isinstance(content, str) and content:
        return content
    return None

  def _terminate_process(self, process: subprocess.Popen[str]) -> None:
    process.terminate()
    process.wait()

  def _close_pipe(self, pipe: Any) -> None:
    if pipe is not None:
      pipe.close()
