import io
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from merceka_core.agent import (
  AgentComplete,
  AgentProfile,
  AgentRawProviderEvent,
  AgentRequest,
  AgentTextDelta,
  ProviderFailure,
)
from merceka_core.agents import _process
from merceka_core.agents.pi import PiAgentProvider

RUN = "merceka_core.agents._process.run"


@pytest.fixture(autouse=True)
def _no_real_processes(monkeypatch: pytest.MonkeyPatch) -> None:
  """No test here may start the real CLI or signal a real process group."""

  def refuse(*args, **_kwargs):
    raise AssertionError(f"unit test tried to launch a real CLI: {args[:1]}")

  monkeypatch.setattr(subprocess, "Popen", refuse)
  monkeypatch.setattr(os, "killpg", lambda *_: None)


def _request(root: Path, profile: AgentProfile = AgentProfile.READ_ONLY) -> AgentRequest:
  return AgentRequest(
    message="Find the thesis",
    system_prompt="Read only.",
    roots=(root,),
    profile=profile,
  )


# `pi --mode json` wire format, from pi 0.85.1 docs/json.md and pi-ai's types:
# message_update records carry only the delta, message_end the whole message.
SESSION = {"type": "session", "version": 3, "id": "uuid", "timestamp": "t", "cwd": "/root"}
USAGE = {"input": 10, "output": 4, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 14}


def _assistant(text: str, stop_reason: str = "stop", error: str | None = None) -> dict:
  message = {
    "role": "assistant",
    "content": [{"type": "text", "text": text}] if text else [],
    "stopReason": stop_reason,
    "usage": USAGE,
  }
  if error is not None:
    message["errorMessage"] = error
  return message


def _text_delta(delta: str) -> dict:
  return {
    "type": "message_update",
    "usage": USAGE,
    "assistantMessageEvent": {"type": "text_delta", "contentIndex": 0, "delta": delta},
  }


def _events(*assistant_turns: tuple[list[str], dict]) -> list[dict]:
  """A session answering in ``assistant_turns``: (text deltas, final message) each."""
  events: list[dict] = [SESSION, {"type": "agent_start"}]
  for deltas, final in assistant_turns:
    events += [
      {"type": "turn_start"},
      {"type": "message_start", "message": {"role": "assistant", "content": []}},
      *[_text_delta(delta) for delta in deltas],
      {"type": "message_end", "message": final},
      {"type": "turn_end", "message": final, "toolResults": []},
    ]
  events.append({"type": "agent_end", "messages": [final for _, final in assistant_turns]})
  return events


def _retried_then_answered(answer: str) -> list[dict]:
  """pi's auto-retry: a failed run, auto_retry_start, then a second run that answers."""
  failed = _events(([], _assistant("", stop_reason="error", error="503 overloaded")))
  retry = {
    "type": "auto_retry_start", "attempt": 1, "maxAttempts": 3, "delayMs": 2000,
    "errorMessage": "503 overloaded",
  }
  answered = _events(([answer], _assistant(answer)))[1:]  # one session header per process
  return [*failed, retry, *answered, {"type": "auto_retry_end", "success": True, "attempt": 1}]


def _stdout(events: list[dict]) -> str:
  return "".join(json.dumps(event) + "\n" for event in events)


class FakePiProcess:
  """A `pi -p --mode json` child replaying ``events`` and then exiting ``returncode``."""

  pid = 424242

  def __init__(self, events: list[dict], returncode: int = 0, stderr: str = ""):
    self.stdin = io.StringIO()
    self.stdout = io.StringIO(_stdout(events))
    self.stderr = io.StringIO(stderr)
    self.returncode: int | None = None
    self._exit_code = returncode

  def poll(self):
    return self.returncode

  def wait(self, timeout=None):
    del timeout  # accepted like Popen.wait; the fake exits at once
    self.returncode = self._exit_code
    return self.returncode


async def _stream(tmp_path: Path, monkeypatch, events, returncode=0, stderr="") -> list:
  process = FakePiProcess(events, returncode, stderr)
  monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
  provider = PiAgentProvider(model="gemini-flash-latest")
  return [event async for event in provider.stream(_request(tmp_path))]


def _completed(stdout: str, returncode: int = 0, stderr: str = ""):
  return patch(
    RUN,
    new_callable=AsyncMock,
    return_value=subprocess.CompletedProcess(["pi"], returncode, stdout=stdout, stderr=stderr),
  )


@pytest.mark.asyncio
async def test_run_invokes_pi_read_only_json_no_session(tmp_path: Path):
  stdout = _stdout(_events((["the ", "answer"], _assistant("the answer"))))

  provider = PiAgentProvider(model="gemini-flash-latest")
  with _completed(stdout) as mock_run:
    result = await provider.run(_request(tmp_path))

  cmd = mock_run.call_args.args[0]
  assert cmd[:2] == ["pi", "-p"]
  assert ["--mode", "json"] == cmd[cmd.index("--mode"):cmd.index("--mode") + 2]
  assert "--no-session" in cmd
  assert ["--model", "gemini-flash-latest"] == cmd[cmd.index("--model"):cmd.index("--model") + 2]
  assert ["--tools", "read,grep,find,ls"] == cmd[cmd.index("--tools"):cmd.index("--tools") + 2]
  assert "--provider" not in cmd
  assert "Read only." in mock_run.call_args.kwargs["input"]
  assert "Find the thesis" in mock_run.call_args.kwargs["input"]
  assert "read-only profile" in mock_run.call_args.kwargs["input"]
  assert mock_run.call_args.kwargs["cwd"] == str(tmp_path.resolve())
  assert result.text == "the answer"
  assert result.raw_events[0].provider == "pi"


@pytest.mark.asyncio
async def test_run_maps_write_profile_to_write_tools(tmp_path: Path):
  provider = PiAgentProvider(model="gemini-flash-latest")
  with _completed("") as mock_run:
    await provider.run(_request(tmp_path, profile=AgentProfile.WRITE))

  cmd = mock_run.call_args.args[0]
  assert ["--tools", "read,grep,find,ls,bash,edit,write"] == cmd[cmd.index("--tools"):cmd.index("--tools") + 2]
  assert "write profile" in mock_run.call_args.kwargs["input"]


@pytest.mark.asyncio
async def test_run_passes_provider_when_set(tmp_path: Path):
  provider = PiAgentProvider(model="anthropic/claude", provider="anthropic")
  with _completed("") as mock_run:
    await provider.run(_request(tmp_path))

  cmd = mock_run.call_args.args[0]
  assert ["--provider", "anthropic"] == cmd[cmd.index("--provider"):cmd.index("--provider") + 2]


@pytest.mark.asyncio
async def test_run_answer_is_the_final_assistant_message(tmp_path: Path):
  # After a tool round the answer is the last assistant message, not every
  # delta of the session joined together.
  stdout = _stdout(_events(
    (["I'll read ", "the index."], _assistant("I'll read the index.", stop_reason="toolUse")),
    (["Page ", "3."], _assistant("Page 3.")),
  ))

  with _completed(stdout):
    result = await PiAgentProvider(model="gemini-flash-latest").run(_request(tmp_path))

  assert result.text == "Page 3."


@pytest.mark.asyncio
async def test_run_error_stop_reason_is_a_provider_failure(tmp_path: Path):
  stdout = _stdout(_events(([], _assistant("", stop_reason="error", error="429 quota exceeded"))))

  with _completed(stdout):
    with pytest.raises(ProviderFailure, match="429 quota exceeded"):
      await PiAgentProvider(model="gemini-flash-latest").run(_request(tmp_path))


@pytest.mark.asyncio
async def test_run_error_recovered_by_a_retry_is_not_a_failure(tmp_path: Path):
  with _completed(_stdout(_retried_then_answered("Page 3."))):
    result = await PiAgentProvider(model="gemini-flash-latest").run(_request(tmp_path))

  assert result.text == "Page 3."


@pytest.mark.asyncio
async def test_run_raises_provider_failure_on_nonzero_exit(tmp_path: Path):
  provider = PiAgentProvider(model="gemini-flash-latest")
  with _completed("", returncode=1, stderr="nope"):
    with pytest.raises(ProviderFailure, match="Pi failed"):
      await provider.run(_request(tmp_path))


@pytest.mark.asyncio
async def test_stream_yields_text_deltas_and_completes_with_the_answer(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  tool_call = {
    "type": "message_update",
    "usage": USAGE,
    "assistantMessageEvent": {
      "type": "toolcall_start", "contentIndex": 1, "id": "call_1", "toolName": "read",
    },
  }
  events = _events((["Hello "], _assistant("Hello world")))
  events.insert(5, tool_call)
  events.insert(6, _text_delta("world"))
  events.insert(8, {
    "type": "tool_execution_end", "toolCallId": "call_1", "toolName": "read",
    "result": {}, "isError": False,
  })

  result = await _stream(tmp_path, monkeypatch, events)

  assert any(isinstance(event, AgentRawProviderEvent) for event in result)
  assert [event.content for event in result if isinstance(event, AgentTextDelta)] == ["Hello ", "world"]
  assert isinstance(result[-1], AgentComplete)
  assert result[-1].result.text == "Hello world"


@pytest.mark.asyncio
async def test_stream_does_not_yield_thinking_as_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  events = _events((["Answer."], _assistant("Answer.")))
  events.insert(4, {
    "type": "message_update",
    "usage": USAGE,
    "assistantMessageEvent": {"type": "thinking_delta", "contentIndex": 0, "delta": "hmm"},
  })

  result = await _stream(tmp_path, monkeypatch, events)

  assert [event.content for event in result if isinstance(event, AgentTextDelta)] == ["Answer."]


@pytest.mark.asyncio
async def test_stream_error_stop_reason_is_a_provider_failure(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  # pi exits 0 in json mode even when the model call failed.
  events = _events(([], _assistant("", stop_reason="error", error="Request aborted")))

  with pytest.raises(ProviderFailure, match="Request aborted"):
    await _stream(tmp_path, monkeypatch, events)


@pytest.mark.asyncio
async def test_stream_error_recovered_by_a_retry_completes_with_the_answer(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  result = await _stream(tmp_path, monkeypatch, _retried_then_answered("Page 3."))

  assert isinstance(result[-1], AgentComplete)
  assert result[-1].result.text == "Page 3."


@pytest.mark.asyncio
async def test_stream_raises_on_nonzero_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  with pytest.raises(ProviderFailure, match="stream failed"):
    await _stream(tmp_path, monkeypatch, [], returncode=1, stderr="stream failed")


def test_malformed_json_line_becomes_raw_event():
  event = _process.raw_event_from_line("not json", "pi")

  assert event.provider == "pi"
  assert event.event_type == "malformed_json"
  assert event.payload["line"] == "not json"
