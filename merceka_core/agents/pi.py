from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from merceka_core._env import scrubbed_env
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

PI_PROVIDER = "pi"
PI_TIMEOUT_SECONDS = 300
READ_ONLY_TOOLS = ("read", "grep", "find", "ls")
WRITE_TOOLS = ("read", "grep", "find", "ls", "bash", "edit", "write")


@dataclass(frozen=True)
class PiAgentProvider:
  model: str = "gemini-flash-latest"
  provider: str | None = None
  pi_binary: str = "pi"
  timeout_seconds: int = PI_TIMEOUT_SECONDS

  async def run(self, request: AgentRequest) -> AgentResult:
    result = await _process.run(
      self._command(request),
      input=self._prompt(request),
      timeout=self.timeout_seconds,
      cwd=str(request.roots[0]),
      env=scrubbed_env(),
      label="Pi",
    )
    raw_events = tuple(_process.raw_events_from_stdout(result.stdout, PI_PROVIDER))
    if result.returncode != 0:
      message = result.stderr.strip() or result.stdout.strip() or "unknown provider error"
      raise ProviderFailure(f"Pi failed with exit {result.returncode}: {message}")
    text = self._final_text(raw_events)
    if not raw_events:
      raw_events = (
        RawProviderEvent(
          provider=PI_PROVIDER,
          event_type="result",
          payload={"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode},
        ),
      )
    return AgentResult(text=text, raw_events=raw_events)

  def stream(self, request: AgentRequest) -> AsyncIterator[AgentStreamEvent]:
    return self._stream(request)

  async def _stream(self, request: AgentRequest) -> AsyncIterator[AgentStreamEvent]:
    stream = _process.Stream(
      self._command(request),
      cwd=str(request.roots[0]),
      env=scrubbed_env(),
      label="Pi stream",
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

        raw_event = _process.raw_event_from_line(line, PI_PROVIDER)
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
        raise ProviderFailure(f"Pi stream failed with exit {returncode}: {message}")
      text = self._final_text(tuple(raw_events)) or "".join(text_chunks)
      yield AgentComplete(result=AgentResult(text=text, raw_events=tuple(raw_events)))
    finally:
      await stream.close()

  def _command(self, request: AgentRequest) -> list[str]:
    cmd = [self.pi_binary, "-p", "--mode", "json", "--no-session", "--model", self.model]
    if self.provider:
      cmd.extend(["--provider", self.provider])
    tools = WRITE_TOOLS if request.profile == AgentProfile.WRITE else READ_ONLY_TOOLS
    cmd.extend(["--tools", ",".join(tools)])
    return cmd

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
    event_type = payload.get("type")
    if isinstance(event_type, str) and event_type.endswith((".output_text.delta", "message_delta")):
      delta = payload.get("delta")
      if isinstance(delta, str) and delta:
        return delta
    return None

  def _final_text(self, raw_events: tuple[RawProviderEvent, ...]) -> str:
    final = ""
    chunks: list[str] = []
    for event in raw_events:
      payload = event.payload
      if not isinstance(payload, dict):
        continue
      text = self._text_delta_from_payload(payload)
      if text is not None:
        chunks.append(text)
      candidate = payload.get("final_text")
      if isinstance(candidate, str) and candidate:
        final = candidate
    return final or "".join(chunks)
