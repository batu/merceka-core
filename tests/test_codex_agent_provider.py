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
from merceka_core.agents.codex import CodexAgentProvider


RUN = "merceka_core.agents._process.run"


@pytest.fixture(autouse=True)
def _no_real_processes(monkeypatch: pytest.MonkeyPatch) -> None:
  """No test here may start the real CLI or signal a real process group."""

  def refuse(*args, **_kwargs):
    raise AssertionError(f"unit test tried to launch a real CLI: {args[:1]}")

  monkeypatch.setattr(subprocess, "Popen", refuse)
  monkeypatch.setattr(os, "killpg", lambda *_: None)


def _request(root: Path) -> AgentRequest:
  return AgentRequest(message="Find the thesis", system_prompt="Read only.", roots=(root,))


@pytest.mark.asyncio
async def test_run_maps_write_profile_to_workspace_write_sandbox(tmp_path: Path):
  def fake_run(cmd, **kwargs):
    Path(cmd[cmd.index("--output-last-message") + 1]).write_text("edited", encoding="utf-8")
    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

  provider = CodexAgentProvider(model="openai/gpt-test")
  request = AgentRequest(
    message="Edit the file",
    system_prompt="You may write.",
    roots=(tmp_path,),
    profile=AgentProfile.WRITE,
  )
  with patch(RUN, new_callable=AsyncMock, side_effect=fake_run) as mock_run:
    await provider.run(request)

  cmd = mock_run.call_args.args[0]
  assert ["--sandbox", "workspace-write"] == cmd[cmd.index("--sandbox"):cmd.index("--sandbox") + 2]
  assert "read-only" not in cmd
  assert "write profile" in mock_run.call_args.kwargs["input"]


@pytest.mark.asyncio
async def test_run_invokes_codex_exec_read_only_with_output_file(tmp_path: Path):
  def fake_run(cmd, **kwargs):
    output_path = Path(cmd[cmd.index("--output-last-message") + 1])
    output_path.write_text("final answer", encoding="utf-8")
    return subprocess.CompletedProcess(cmd, 0, stdout='{"type":"done"}\n', stderr="")

  provider = CodexAgentProvider(model="openai/gpt-test")
  with patch(RUN, new_callable=AsyncMock, side_effect=fake_run) as mock_run:
    result = await provider.run(_request(tmp_path))

  cmd = mock_run.call_args.args[0]
  assert cmd[:2] == ["codex", "exec"]
  assert ["--model", "openai/gpt-test"] == cmd[2:4]
  assert "--json" in cmd
  assert ["--sandbox", "read-only"] == cmd[cmd.index("--sandbox"):cmd.index("--sandbox") + 2]
  assert ["--cd", str(tmp_path.resolve())] == cmd[cmd.index("--cd"):cmd.index("--cd") + 2]
  assert "Read only." in mock_run.call_args.kwargs["input"]
  assert "Find the thesis" in mock_run.call_args.kwargs["input"]
  assert result.text == "final answer"
  assert result.raw_events[0].provider == "codex"


@pytest.mark.asyncio
async def test_default_high_alias_uses_account_default_model_with_high_effort(tmp_path: Path):
  def fake_run(cmd, **kwargs):
    Path(cmd[cmd.index("--output-last-message") + 1]).write_text("ok", encoding="utf-8")
    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

  provider = CodexAgentProvider(model="gpt-5.5-high")
  with patch(RUN, new_callable=AsyncMock, side_effect=fake_run) as mock_run:
    await provider.run(_request(tmp_path))

  cmd = mock_run.call_args.args[0]
  assert "--model" not in cmd
  assert ["-c", 'model_reasoning_effort="high"'] == cmd[2:4]


@pytest.mark.asyncio
async def test_run_adds_secondary_roots(tmp_path: Path):
  first = tmp_path / "first"
  second = tmp_path / "second"
  first.mkdir()
  second.mkdir()

  def fake_run(cmd, **kwargs):
    Path(cmd[cmd.index("--output-last-message") + 1]).write_text("ok", encoding="utf-8")
    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

  provider = CodexAgentProvider(model="gpt-5.5-high")
  with patch(RUN, new_callable=AsyncMock, side_effect=fake_run) as mock_run:
    await provider.run(AgentRequest(message="Q", system_prompt="P", roots=(first, second)))

  cmd = mock_run.call_args.args[0]
  assert ["--add-dir", str(second.resolve())] == cmd[cmd.index("--add-dir"):cmd.index("--add-dir") + 2]


@pytest.mark.asyncio
async def test_run_raises_provider_failure_on_nonzero_exit(tmp_path: Path):
  provider = CodexAgentProvider(model="gpt-5.5-high")
  with patch(RUN, new_callable=AsyncMock, return_value=subprocess.CompletedProcess(["codex"], 1, stdout="", stderr="nope")):
    with pytest.raises(ProviderFailure, match="Codex failed"):
      await provider.run(_request(tmp_path))


class FakeCodexProcess:
  """A codex exec child whose stdout replays ``lines`` and then exits ``returncode``."""

  pid = 424242

  def __init__(self, lines: list[dict], returncode: int = 0, stderr: str = ""):
    self.stdin = io.StringIO()
    self.stdout = io.StringIO("".join(json.dumps(line) + "\n" for line in lines))
    self.stderr = io.StringIO(stderr)
    self.returncode: int | None = None
    self._exit_code = returncode

  def poll(self):
    return self.returncode

  def wait(self, timeout=None):

    del timeout  # accepted like Popen.wait; the fake exits at once
    self.returncode = self._exit_code
    return self.returncode


async def _stream(tmp_path: Path, monkeypatch, lines, returncode=0, stderr="") -> list:
  process = FakeCodexProcess(lines, returncode, stderr)
  monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
  provider = CodexAgentProvider(model="gpt-5.5-high")
  events = []
  try:
    async for event in provider.stream(_request(tmp_path)):
      events.append(event)
  except ProviderFailure as failure:
    failure.events = events  # type: ignore[attr-defined]
    raise
  return events


def _text(events) -> list[str]:
  return [event.content for event in events if isinstance(event, AgentTextDelta)]


# `codex exec --json` wire format, from the Codex docs and codex-rs exec_events.rs.
THREAD_STARTED = {"type": "thread.started", "thread_id": "0199a213-81c0-7800-8aa1-bbab2a035a53"}
TURN_STARTED = {"type": "turn.started"}
TURN_COMPLETED = {
  "type": "turn.completed",
  "usage": {"input_tokens": 24763, "cached_input_tokens": 24448, "output_tokens": 122},
}


def _agent_message(text: str, item_id: str = "item_3") -> dict:
  return {"type": "item.completed", "item": {"id": item_id, "type": "agent_message", "text": text}}


@pytest.mark.asyncio
async def test_stream_yields_the_agent_message_as_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  events = await _stream(tmp_path, monkeypatch, [
    THREAD_STARTED,
    TURN_STARTED,
    {"type": "item.completed", "item": {"id": "item_0", "type": "reasoning", "text": "**Scanning**"}},
    {"type": "item.started", "item": {
      "id": "item_1", "type": "command_execution", "command": "bash -lc ls",
      "aggregated_output": "", "exit_code": None, "status": "in_progress",
    }},
    {"type": "item.completed", "item": {
      "id": "item_1", "type": "command_execution", "command": "bash -lc ls",
      "aggregated_output": "docs\nsrc\n", "exit_code": 0, "status": "completed",
    }},
    _agent_message("Repo contains docs, sdk, and examples directories."),
    TURN_COMPLETED,
  ])

  assert any(isinstance(event, AgentRawProviderEvent) for event in events)
  assert _text(events) == ["Repo contains docs, sdk, and examples directories."]
  assert isinstance(events[-1], AgentComplete)
  assert events[-1].result.text == "Repo contains docs, sdk, and examples directories."


@pytest.mark.asyncio
async def test_stream_completes_with_the_last_agent_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  # Like run()'s --output-last-message: the final agent message is the answer.
  events = await _stream(tmp_path, monkeypatch, [
    THREAD_STARTED,
    _agent_message("I'll read the index first.", "item_1"),
    _agent_message("The thesis is on page 3.", "item_4"),
    TURN_COMPLETED,
  ])

  assert _text(events) == ["I'll read the index first.", "The thesis is on page 3."]
  assert events[-1].result.text == "The thesis is on page 3."


@pytest.mark.asyncio
async def test_stream_error_event_is_a_provider_failure_not_text(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  with pytest.raises(ProviderFailure, match="stream disconnected before completion") as failure:
    await _stream(tmp_path, monkeypatch, [
      THREAD_STARTED,
      {"type": "error", "message": "stream disconnected before completion"},
    ], returncode=1)

  assert _text(failure.value.events) == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_stream_turn_failed_is_a_provider_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  with pytest.raises(ProviderFailure, match="model response stream ended unexpectedly"):
    await _stream(tmp_path, monkeypatch, [
      THREAD_STARTED,
      TURN_STARTED,
      {"type": "turn.failed", "error": {"message": "model response stream ended unexpectedly"}},
    ], returncode=1)


@pytest.mark.asyncio
async def test_stream_error_without_a_completed_turn_fails_even_on_exit_zero(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  with pytest.raises(ProviderFailure, match="usage limit reached"):
    await _stream(tmp_path, monkeypatch, [
      THREAD_STARTED,
      {"type": "error", "message": "usage limit reached"},
    ])


@pytest.mark.asyncio
async def test_stream_recovered_reconnect_notice_is_neither_text_nor_failure(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  # Codex reports transient reconnects as error events and then carries on.
  events = await _stream(tmp_path, monkeypatch, [
    THREAD_STARTED,
    {"type": "error", "message": "Reconnecting... 1/5 (stream disconnected before completion)"},
    _agent_message("Done."),
    TURN_COMPLETED,
  ])

  assert _text(events) == ["Done."]
  assert events[-1].result.text == "Done."


@pytest.mark.asyncio
async def test_stream_error_after_the_turn_completed_is_a_provider_failure(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  with pytest.raises(ProviderFailure, match="broken pipe"):
    await _stream(tmp_path, monkeypatch, [
      THREAD_STARTED,
      _agent_message("Done."),
      TURN_COMPLETED,
      {"type": "error", "message": "stream error: broken pipe"},
    ])


@pytest.mark.asyncio
async def test_stream_error_item_is_a_warning_not_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  events = await _stream(tmp_path, monkeypatch, [
    THREAD_STARTED,
    {"type": "item.completed", "item": {"id": "item_9", "type": "error", "message": "command output truncated"}},
    _agent_message("Done."),
    TURN_COMPLETED,
  ])

  assert _text(events) == ["Done."]
