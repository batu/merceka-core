from __future__ import annotations

import asyncio
import json
import os
import subprocess
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

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

CLAUDE_CODE_PROVIDER = "claude_code"
CLAUDE_CODE_TIMEOUT_SECONDS = 120
READ_ONLY_TOOLS = ("Read", "Grep", "Glob")
WRITE_TOOLS = ("Read", "Grep", "Glob", "Edit", "Write", "Bash")


@dataclass(frozen=True)
class ClaudeCodeAgentProvider:
  model: str
  claude_binary: str = "claude"
  timeout_seconds: int = CLAUDE_CODE_TIMEOUT_SECONDS

  async def run(self, request: AgentRequest) -> AgentResult:
    return await asyncio.to_thread(self._run_sync, request)

  def stream(self, request: AgentRequest) -> AsyncIterator[AgentStreamEvent]:
    return self._stream(request)

  def _run_sync(self, request: AgentRequest) -> AgentResult:
    cmd = self._command(request, stream=False)
    result = subprocess.run(
      cmd,
      input=request.message,
      capture_output=True,
      text=True,
      timeout=self.timeout_seconds,
      env=self._env(),
      cwd=str(request.roots[0]),
    )
    raw_event = RawProviderEvent(
      provider=CLAUDE_CODE_PROVIDER,
      event_type="result",
      payload={
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
      },
    )
    if result.returncode != 0:
      message = result.stderr.strip() or result.stdout.strip() or "unknown provider error"
      raise ProviderFailure(f"Claude Code failed with exit {result.returncode}: {message}")
    return AgentResult(text=result.stdout.strip(), raw_events=(raw_event,))

  async def _stream(self, request: AgentRequest) -> AsyncIterator[AgentStreamEvent]:
    cmd = self._command(request, stream=True)
    process = subprocess.Popen(
      cmd,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      bufsize=1,
      env=self._env(),
      cwd=str(request.roots[0]),
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
      raise ProviderFailure("Claude Code stream did not expose stdio pipes")

    process.stdin.write(request.message)
    process.stdin.close()

    raw_events: list[RawProviderEvent] = []
    text_chunks: list[str] = []
    completed = False
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
        if not isinstance(payload, dict):
          continue

        text = self._text_delta_from_payload(payload)
        if text is not None:
          text_chunks.append(text)
          yield AgentTextDelta(content=text)

        if payload.get("type") == "result":
          completed = True
          break

      returncode = await asyncio.to_thread(process.wait)
      stderr = await asyncio.to_thread(process.stderr.read)
      if returncode != 0:
        message = stderr.strip() or f"exit {returncode}"
        raise ProviderFailure(f"Claude Code stream failed with exit {returncode}: {message}")
      if not completed:
        completion_event = RawProviderEvent(
          provider=CLAUDE_CODE_PROVIDER,
          event_type="stream_closed",
          payload={"returncode": returncode, "stderr": stderr},
        )
        raw_events.append(completion_event)
        yield AgentRawProviderEvent(raw_event=completion_event)
      yield AgentComplete(result=AgentResult(text="".join(text_chunks), raw_events=tuple(raw_events)))
    except GeneratorExit:
      self._terminate_process(process)
      raise
    except asyncio.CancelledError:
      self._terminate_process(process)
      raise
    finally:
      if not completed and process.returncode is None:
        self._terminate_process(process)
      self._close_pipe(process.stdout)
      self._close_pipe(process.stderr)

  def _command(self, request: AgentRequest, *, stream: bool) -> list[str]:
    cmd = [self.claude_binary, "-p", "--model", self.model]
    if stream:
      cmd.extend([
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
      ])
    if request.profile == AgentProfile.WRITE:
      cmd.extend(["--permission-mode", "acceptEdits"])
    if request.system_prompt:
      cmd.extend(["--system-prompt", request.system_prompt])
    for root in request.roots:
      cmd.extend(["--add-dir", str(root)])
    tools = WRITE_TOOLS if request.profile == AgentProfile.WRITE else READ_ONLY_TOOLS
    cmd.extend(["--allowedTools", ",".join(tools)])
    return cmd

  def _raw_event_from_line(self, line: str) -> RawProviderEvent:
    try:
      payload: Any = json.loads(line)
    except json.JSONDecodeError as exc:
      return RawProviderEvent(
        provider=CLAUDE_CODE_PROVIDER,
        event_type="malformed_json",
        payload={"line": line, "error": str(exc)},
      )
    event_type = str(payload.get("type", "raw")) if isinstance(payload, dict) else "raw"
    return RawProviderEvent(provider=CLAUDE_CODE_PROVIDER, event_type=event_type, payload=payload)

  def _text_delta_from_payload(self, payload: dict[str, Any]) -> str | None:
    if payload.get("type") != "stream_event":
      return None
    event = payload.get("event")
    if not isinstance(event, dict) or event.get("type") != "content_block_delta":
      return None
    delta = event.get("delta")
    if not isinstance(delta, dict) or delta.get("type") != "text_delta":
      return None
    text = delta.get("text")
    return text if isinstance(text, str) else None

  def _env(self) -> dict[str, str]:
    return {**os.environ, "ANTHROPIC_API_KEY": ""}

  def _terminate_process(self, process: subprocess.Popen[str]) -> None:
    process.terminate()
    process.wait()

  def _close_pipe(self, pipe: Any) -> None:
    if pipe is not None:
      pipe.close()
