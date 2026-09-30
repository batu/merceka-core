from __future__ import annotations

import io
import json
import os
import signal
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from merceka_core.agent import (
  AgentComplete,
  AgentProfile,
  AgentRawProviderEvent,
  AgentRequest,
  AgentTextDelta,
  ProviderFailure,
)
from merceka_core.agents.claude_code import ClaudeCodeAgentProvider


RUN = "merceka_core.agents._process.run"


@pytest.fixture(autouse=True)
def signals(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
  """No test here may start the real CLI or signal a real process group."""

  def refuse(*args, **_kwargs):
    raise AssertionError(f"unit test tried to launch a real CLI: {args[:1]}")

  sent: list[tuple[int, int]] = []
  monkeypatch.setattr(subprocess, "Popen", refuse)
  monkeypatch.setattr(os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))
  return sent


def _request(root: Path) -> AgentRequest:
  return AgentRequest(
    message="What is in the book?",
    system_prompt="Read before answering.",
    roots=(root,),
  )


@pytest.mark.asyncio
async def test_claude_agent_run_builds_read_only_command(tmp_path: Path):
  provider = ClaudeCodeAgentProvider(model="sonnet")

  with patch(RUN, new_callable=AsyncMock) as mock_run:
    mock_run.return_value = MagicMock(returncode=0, stdout="Answer\n", stderr="")
    result = await provider.run(_request(tmp_path))

  cmd = mock_run.call_args.args[0]
  assert cmd == [
    "claude",
    "-p",
    "--model",
    "sonnet",
    *[
    "--permission-mode",
    "dontAsk",
    "--setting-sources",
    "user",
    "--strict-mcp-config",
    "--disallowedTools",
    "mcp__*",
    "--tools",
    "Read,Grep,Glob",
  ],
    "--append-system-prompt",
    "Read before answering.",
    "--add-dir",
    str(tmp_path),
  ]
  assert mock_run.call_args.kwargs["input"] == "What is in the book?"
  assert mock_run.call_args.kwargs["cwd"] == str(tmp_path)
  assert mock_run.call_args.kwargs["env"]["ANTHROPIC_API_KEY"] == ""
  assert result.text == "Answer"
  assert result.raw_events[0].provider == "claude_code"


@pytest.mark.asyncio
async def test_claude_agent_run_builds_write_command(tmp_path: Path):
  provider = ClaudeCodeAgentProvider(model="sonnet")
  request = AgentRequest(
    message="Edit the file",
    system_prompt="You may write.",
    roots=(tmp_path,),
    profile=AgentProfile.WRITE,
  )

  with patch(RUN, new_callable=AsyncMock) as mock_run:
    mock_run.return_value = MagicMock(returncode=0, stdout="Done\n", stderr="")
    await provider.run(request)

  cmd = mock_run.call_args.args[0]
  assert cmd == [
    "claude",
    "-p",
    "--model",
    "sonnet",
    "--permission-mode",
    "acceptEdits",
    "--append-system-prompt",
    "You may write.",
    "--add-dir",
    str(tmp_path),
    "--allowedTools",
    "Read,Grep,Glob,Edit,Write,Bash",
  ]


@pytest.mark.asyncio
async def test_claude_read_only_command_is_locked_down(tmp_path: Path):
  """READ_ONLY must restrict the tool set, not just pre-approve Read/Grep/Glob.

  The historical command passed only ``--allowedTools Read,Grep,Glob``. That
  pre-approves those tools but leaves Bash, Edit and Write available under the
  user's ``defaultMode`` (``auto`` on both machines).
  """
  provider = ClaudeCodeAgentProvider(model="sonnet")

  with patch(RUN, new_callable=AsyncMock) as mock_run:
    mock_run.return_value = MagicMock(returncode=0, stdout="Answer\n", stderr="")
    await provider.run(_request(tmp_path))

  cmd = mock_run.call_args.args[0]
  assert cmd[cmd.index("--tools") + 1] == "Read,Grep,Glob"
  assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
  assert cmd[cmd.index("--setting-sources") + 1] == "user"
  assert cmd[cmd.index("--disallowedTools") + 1] == "mcp__*"
  assert "--strict-mcp-config" in cmd
  # Read/Grep/Glob are not pre-approved: under dontAsk they stay confined to the
  # working directories instead of reaching ~/.ssh or a .env elsewhere.
  assert "--allowedTools" not in cmd
  for tool in ("Bash", "Edit", "Write"):
    assert tool not in cmd[cmd.index("--tools") + 1].split(",")


@pytest.mark.asyncio
async def test_claude_agent_run_passes_multiple_roots(tmp_path: Path):
  first_root = tmp_path / "one"
  second_root = tmp_path / "two"
  first_root.mkdir()
  second_root.mkdir()
  provider = ClaudeCodeAgentProvider(model="sonnet")

  with patch(RUN, new_callable=AsyncMock) as mock_run:
    mock_run.return_value = MagicMock(returncode=0, stdout="Answer", stderr="")
    await provider.run(AgentRequest(message="Q", system_prompt="P", roots=(first_root, second_root)))

  cmd = mock_run.call_args.args[0]
  assert cmd.count("--add-dir") == 2
  assert cmd[cmd.index("--add-dir") + 1] == str(first_root)
  assert cmd[cmd.index("--add-dir", cmd.index("--add-dir") + 1) + 1] == str(second_root)
  assert mock_run.call_args.kwargs["cwd"] == str(first_root)


@pytest.mark.asyncio
async def test_claude_agent_run_raises_on_nonzero_exit(tmp_path: Path):
  provider = ClaudeCodeAgentProvider(model="sonnet")

  with patch(RUN, new_callable=AsyncMock) as mock_run:
    mock_run.return_value = MagicMock(returncode=2, stdout="partial", stderr="boom")
    with pytest.raises(ProviderFailure, match="Claude Code failed"):
      await provider.run(_request(tmp_path))


class FakePopen:
  pid = 424242

  def __init__(self, lines: list[str], returncode: int = 0, stderr: str = ""):
    self.stdin = io.StringIO()
    self.stdout = io.StringIO("".join(lines))
    self.stderr = io.StringIO(stderr)
    self.returncode: int | None = None
    self._exit_code = returncode
    self.wait_called = False

  def poll(self) -> int | None:
    return self.returncode

  def wait(self, timeout: float | None = None) -> int:
    self.wait_called = True
    self.returncode = self._exit_code
    return self.returncode


def _stream_line(obj: dict) -> str:
  return json.dumps(obj) + "\n"


@pytest.mark.asyncio
async def test_claude_agent_stream_parses_text_raw_and_completion(tmp_path: Path):
  process = FakePopen([
    _stream_line({
      "type": "stream_event",
      "event": {
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": "Hello "},
      },
    }),
    _stream_line({
      "type": "stream_event",
      "event": {
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": "world"},
      },
    }),
    _stream_line({"type": "result", "subtype": "success"}),
  ])
  provider = ClaudeCodeAgentProvider(model="sonnet")

  with patch("subprocess.Popen", return_value=process):
    events = [event async for event in provider.stream(_request(tmp_path))]

  text = "".join(event.content for event in events if isinstance(event, AgentTextDelta))
  raw_events = [event for event in events if isinstance(event, AgentRawProviderEvent)]
  completions = [event for event in events if isinstance(event, AgentComplete)]
  assert text == "Hello world"
  assert len(raw_events) == 3
  assert completions[0].result.text == "Hello world"
  assert process.wait_called is True


@pytest.mark.asyncio
async def test_claude_agent_stream_nonzero_exit_raises_failure(tmp_path: Path):
  process = FakePopen([], returncode=1, stderr="stream failed")
  provider = ClaudeCodeAgentProvider(model="sonnet")

  with patch("subprocess.Popen", return_value=process):
    with pytest.raises(ProviderFailure, match="stream failed"):
      [event async for event in provider.stream(_request(tmp_path))]


@pytest.mark.asyncio
async def test_claude_agent_stream_malformed_json_becomes_raw_event(tmp_path: Path):
  process = FakePopen(["not json\n", _stream_line({"type": "result"})])
  provider = ClaudeCodeAgentProvider(model="sonnet")

  with patch("subprocess.Popen", return_value=process):
    events = [event async for event in provider.stream(_request(tmp_path))]

  raw_events = [event.raw_event for event in events if isinstance(event, AgentRawProviderEvent)]
  assert raw_events[0].event_type == "malformed_json"
  assert raw_events[0].payload["line"] == "not json"


@pytest.mark.asyncio
async def test_claude_agent_stream_closes_process_when_generator_closes(tmp_path: Path, signals):
  process = FakePopen([
    _stream_line({
      "type": "stream_event",
      "event": {
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": "Hello"},
      },
    }),
    _stream_line({"type": "result"}),
  ])
  provider = ClaudeCodeAgentProvider(model="sonnet")

  with patch("subprocess.Popen", return_value=process):
    stream = provider.stream(_request(tmp_path))
    event = await anext(stream)
    assert isinstance(event, AgentRawProviderEvent)
    await stream.aclose()

  assert (process.pid, signal.SIGTERM) in signals
  assert process.stdout.closed and process.stderr.closed
