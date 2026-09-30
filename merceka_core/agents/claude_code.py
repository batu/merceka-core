from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
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
    result = await _process.run(
      self._command(request, stream=False),
      input=request.message,
      timeout=self.timeout_seconds,
      cwd=str(request.roots[0]),
      env=self._env(),
      label="Claude Code",
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

  def stream(self, request: AgentRequest) -> AsyncIterator[AgentStreamEvent]:
    return self._stream(request)

  async def _stream(self, request: AgentRequest) -> AsyncIterator[AgentStreamEvent]:
    stream = _process.Stream(
      self._command(request, stream=True),
      cwd=str(request.roots[0]),
      env=self._env(),
      timeout=self.timeout_seconds,
      label="Claude Code stream",
    )
    raw_events: list[RawProviderEvent] = []
    text_chunks: list[str] = []
    completed = False
    try:
      await stream.send(request.message)
      while True:
        line = await stream.readline()
        if line == "":
          break
        line = line.strip()
        if not line:
          continue

        raw_event = _process.raw_event_from_line(line, CLAUDE_CODE_PROVIDER)
        raw_events.append(raw_event)
        yield AgentRawProviderEvent(raw_event=raw_event)

        payload = raw_event.payload
        if not isinstance(payload, dict):
          continue

        text = self._text_delta_from_payload(payload)
        if text is not None:
          text_chunks.append(text)
          yield AgentTextDelta(content=text)

        if _cli.is_claude_result_event(payload):
          completed = True
          break

      returncode, stderr = await stream.finish()
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
    finally:
      await stream.close()

  def _command(self, request: AgentRequest, *, stream: bool) -> list[str]:
    tools = WRITE_TOOLS if request.profile == AgentProfile.WRITE else READ_ONLY_TOOLS
    return _cli.claude_command(
      self.model,
      system_prompt=request.system_prompt,
      add_dirs=[str(root) for root in request.roots],
      allowed_tools=tools,
      stream=stream,
      accept_edits=request.profile == AgentProfile.WRITE,
      binary=self.claude_binary,
    )

  def _text_delta_from_payload(self, payload: dict[str, Any]) -> str | None:
    return _cli.claude_stream_text_delta(payload)

  def _env(self) -> dict[str, str]:
    return _cli.claude_env()
