import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from merceka_core.agent import AgentComplete, AgentRawProviderEvent, AgentRequest, AgentTextDelta, ProviderFailure
from merceka_core.agents.codex import CodexAgentProvider


def _request(root: Path) -> AgentRequest:
  return AgentRequest(message="Find the thesis", system_prompt="Read only.", roots=(root,))


@pytest.mark.asyncio
async def test_run_invokes_codex_exec_read_only_with_output_file(tmp_path: Path):
  def fake_run(cmd, *, input, capture_output, text, timeout, cwd):
    output_path = Path(cmd[cmd.index("--output-last-message") + 1])
    output_path.write_text("final answer", encoding="utf-8")
    return subprocess.CompletedProcess(cmd, 0, stdout='{"type":"done"}\n', stderr="")

  provider = CodexAgentProvider(model="openai/gpt-test")
  with patch("subprocess.run", side_effect=fake_run) as mock_run:
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
  def fake_run(cmd, *, input, capture_output, text, timeout, cwd):
    Path(cmd[cmd.index("--output-last-message") + 1]).write_text("ok", encoding="utf-8")
    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

  provider = CodexAgentProvider(model="gpt-5.5-high")
  with patch("subprocess.run", side_effect=fake_run) as mock_run:
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

  def fake_run(cmd, *, input, capture_output, text, timeout, cwd):
    Path(cmd[cmd.index("--output-last-message") + 1]).write_text("ok", encoding="utf-8")
    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

  provider = CodexAgentProvider(model="gpt-5.5-high")
  with patch("subprocess.run", side_effect=fake_run) as mock_run:
    await provider.run(AgentRequest(message="Q", system_prompt="P", roots=(first, second)))

  cmd = mock_run.call_args.args[0]
  assert ["--add-dir", str(second.resolve())] == cmd[cmd.index("--add-dir"):cmd.index("--add-dir") + 2]


@pytest.mark.asyncio
async def test_run_raises_provider_failure_on_nonzero_exit(tmp_path: Path):
  provider = CodexAgentProvider(model="gpt-5.5-high")
  with patch("subprocess.run", return_value=subprocess.CompletedProcess(["codex"], 1, stdout="", stderr="nope")):
    with pytest.raises(ProviderFailure, match="Codex failed"):
      await provider.run(_request(tmp_path))


@pytest.mark.asyncio
async def test_stream_normalizes_json_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  class FakeStdout:
    def __init__(self):
      self.lines = iter([
        json.dumps({"type": "message_delta", "delta": "Hello "}) + "\n",
        json.dumps({"type": "agent_message", "message": "world"}) + "\n",
        "",
      ])

    def readline(self):
      return next(self.lines)

    def close(self):
      return None

  class FakeStdin:
    def write(self, text):
      self.text = text

    def close(self):
      return None

  class FakeStderr:
    def read(self):
      return ""

    def close(self):
      return None

  class FakeProcess:
    def __init__(self, *args, **kwargs):
      self.stdin = FakeStdin()
      self.stdout = FakeStdout()
      self.stderr = FakeStderr()
      self.returncode = 0
      self.terminated = False

    def wait(self):
      return 0

    def terminate(self):
      self.terminated = True

  monkeypatch.setattr(subprocess, "Popen", FakeProcess)

  provider = CodexAgentProvider(model="gpt-5.5-high")
  events = [event async for event in provider.stream(_request(tmp_path))]

  assert any(isinstance(event, AgentRawProviderEvent) for event in events)
  assert [event.content for event in events if isinstance(event, AgentTextDelta)] == ["Hello ", "world"]
  assert isinstance(events[-1], AgentComplete)
  assert events[-1].result.text == "Hello world"
