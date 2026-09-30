from __future__ import annotations

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
from merceka_core.agents import _process

CODEX_PROVIDER = "codex"
CODEX_TIMEOUT_SECONDS = 300
DEFAULT_CODEX_MODEL_ALIASES = {"", "default", "codex-default", "codex-default-high", "gpt-5.5-high"}


@dataclass(frozen=True)
class CodexAgentProvider:
  model: str = "codex-default-high"
  codex_binary: str = "codex"
  timeout_seconds: int = CODEX_TIMEOUT_SECONDS

  async def run(self, request: AgentRequest) -> AgentResult:
    with tempfile.NamedTemporaryFile("r", encoding="utf-8", delete=False) as output_file:
      output_path = Path(output_file.name)
    try:
      cmd = self._command(request, json_output=True)
      cmd.extend(["--output-last-message", str(output_path)])
      result = await _process.run(
        cmd,
        input=self._prompt(request),
        timeout=self.timeout_seconds,
        cwd=str(request.roots[0]),
        env=_cli.codex_env(),
        label="Codex",
      )
      raw_events = tuple(_process.raw_events_from_stdout(result.stdout, CODEX_PROVIDER))
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

  def stream(self, request: AgentRequest) -> AsyncIterator[AgentStreamEvent]:
    return self._stream(request)

  async def _stream(self, request: AgentRequest) -> AsyncIterator[AgentStreamEvent]:
    stream = _process.Stream(
      self._command(request, json_output=True),
      cwd=str(request.roots[0]),
      env=_cli.codex_env(),
      timeout=self.timeout_seconds,
      label="Codex stream",
    )
    raw_events: list[RawProviderEvent] = []
    text_chunks: list[str] = []
    try:
      await stream.send(self._prompt(request))
      while True:
        line = await stream.readline()
        if line == "":
          break
        line = line.strip()
        if not line:
          continue

        raw_event = _process.raw_event_from_line(line, CODEX_PROVIDER)
        raw_events.append(raw_event)
        yield AgentRawProviderEvent(raw_event=raw_event)

        payload = raw_event.payload
        if isinstance(payload, dict):
          text = self._text_delta_from_payload(payload)
          if text is not None:
            text_chunks.append(text)
            yield AgentTextDelta(content=text)

      returncode, stderr = await stream.finish()
      if returncode != 0:
        message = stderr.strip() or f"exit {returncode}"
        raise ProviderFailure(f"Codex stream failed with exit {returncode}: {message}")
      yield AgentComplete(result=AgentResult(text="".join(text_chunks), raw_events=tuple(raw_events)))
    finally:
      await stream.close()

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
