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
    failure = _run_failure(raw_events)
    if failure is not None:
      raise ProviderFailure(f"Pi failed: {failure}")
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
      timeout=self.timeout_seconds,
      label="Pi stream",
    )
    raw_events: list[RawProviderEvent] = []
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
            yield AgentTextDelta(content=text)

      returncode, stderr = await stream.finish()
      if returncode != 0:
        message = stderr.strip() or f"exit {returncode}"
        raise ProviderFailure(f"Pi stream failed with exit {returncode}: {message}")
      failure = _run_failure(tuple(raw_events))
      if failure is not None:
        raise ProviderFailure(f"Pi stream failed: {failure}")
      text = self._final_text(tuple(raw_events))
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
    """Answer text streamed in a ``message_update`` event.

    ``pi --mode json`` sends each streaming step of an assistant message as
    ``message_update`` with ``assistantMessageEvent``; only its ``text_delta``
    steps are answer text (thinking and tool-call deltas are not).
    """
    if payload.get("type") != "message_update":
      return None
    event = payload.get("assistantMessageEvent")
    if not isinstance(event, dict) or event.get("type") != "text_delta":
      return None
    delta = event.get("delta")
    return delta if isinstance(delta, str) and delta else None

  def _final_text(self, raw_events: tuple[RawProviderEvent, ...]) -> str:
    """The text of the last assistant ``message_end``, the authoritative message.

    Falls back to the joined text deltas if no assistant message ended.
    """
    last = _last_assistant_message(raw_events)
    if last is not None:
      return _message_text(last)
    return "".join(
      text
      for event in raw_events
      if isinstance(event.payload, dict)
      and (text := self._text_delta_from_payload(event.payload)) is not None
    )


def _last_assistant_message(raw_events: tuple[RawProviderEvent, ...]) -> dict[str, Any] | None:
  """The message of the last assistant ``message_end`` event, if any."""
  last = None
  for event in raw_events:
    payload = event.payload
    if not isinstance(payload, dict) or payload.get("type") != "message_end":
      continue
    message = payload.get("message")
    if isinstance(message, dict) and message.get("role") == "assistant":
      last = message
  return last


def _message_text(message: dict[str, Any]) -> str:
  content = message.get("content")
  if isinstance(content, str):
    return content
  if not isinstance(content, list):
    return ""
  return "".join(
    part["text"]
    for part in content
    if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
  )


def _run_failure(raw_events: tuple[RawProviderEvent, ...]) -> str | None:
  """Why the run failed, judged the way pi's own print mode judges it.

  pi exits 0 in json mode even when the model call failed: the failure is
  only the last assistant message's ``stopReason``. Only the last message
  counts, because pi retries some errors itself and a later message can
  still answer.
  """
  last = _last_assistant_message(raw_events)
  if last is None or last.get("stopReason") not in ("error", "aborted"):
    return None
  return str(last.get("errorMessage") or f"request {last['stopReason']}")
