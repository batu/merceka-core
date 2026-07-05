import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from merceka_core.agent import (
  AgentComplete,
  AgentProfile,
  AgentRawProviderEvent,
  AgentRequest,
  AgentTextDelta,
  ProviderFailure,
)
from merceka_core.agents.pi import PiAgentProvider


def _request(root: Path, profile: AgentProfile = AgentProfile.READ_ONLY) -> AgentRequest:
  return AgentRequest(
    message="Find the thesis",
    system_prompt="Read only.",
    roots=(root,),
    profile=profile,
  )


@pytest.mark.asyncio
async def test_run_invokes_pi_read_only_json_no_session(tmp_path: Path):
  stdout = (
    json.dumps({"type": "response.output_text.delta", "delta": "the "}) + "\n"
    + json.dumps({"type": "response.output_text.delta", "delta": "answer"}) + "\n"
    + json.dumps({"type": "turn.complete", "final_text": "the answer"}) + "\n"
  )

  provider = PiAgentProvider(model="gemini-flash-latest")
  with patch(
    "subprocess.run",
    return_value=subprocess.CompletedProcess(["pi"], 0, stdout=stdout, stderr=""),
  ) as mock_run:
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
  with patch(
    "subprocess.run",
    return_value=subprocess.CompletedProcess(["pi"], 0, stdout="", stderr=""),
  ) as mock_run:
    await provider.run(_request(tmp_path, profile=AgentProfile.WRITE))

  cmd = mock_run.call_args.args[0]
  assert ["--tools", "read,grep,find,ls,bash,edit,write"] == cmd[cmd.index("--tools"):cmd.index("--tools") + 2]
  assert "write profile" in mock_run.call_args.kwargs["input"]


@pytest.mark.asyncio
async def test_run_passes_provider_when_set(tmp_path: Path):
  provider = PiAgentProvider(model="anthropic/claude", provider="anthropic")
  with patch(
    "subprocess.run",
    return_value=subprocess.CompletedProcess(["pi"], 0, stdout="", stderr=""),
  ) as mock_run:
    await provider.run(_request(tmp_path))

  cmd = mock_run.call_args.args[0]
  assert ["--provider", "anthropic"] == cmd[cmd.index("--provider"):cmd.index("--provider") + 2]


@pytest.mark.asyncio
async def test_run_falls_back_to_joined_deltas_without_final_text(tmp_path: Path):
  stdout = (
    json.dumps({"type": "response.output_text.delta", "delta": "Hello "}) + "\n"
    + json.dumps({"type": "response.output_text.delta", "delta": "world"}) + "\n"
  )
  provider = PiAgentProvider(model="gemini-flash-latest")
  with patch(
    "subprocess.run",
    return_value=subprocess.CompletedProcess(["pi"], 0, stdout=stdout, stderr=""),
  ):
    result = await provider.run(_request(tmp_path))

  assert result.text == "Hello world"


@pytest.mark.asyncio
async def test_run_raises_provider_failure_on_nonzero_exit(tmp_path: Path):
  provider = PiAgentProvider(model="gemini-flash-latest")
  with patch(
    "subprocess.run",
    return_value=subprocess.CompletedProcess(["pi"], 1, stdout="", stderr="nope"),
  ):
    with pytest.raises(ProviderFailure, match="Pi failed"):
      await provider.run(_request(tmp_path))


@pytest.mark.asyncio
async def test_stream_normalizes_json_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  class FakeStdout:
    def __init__(self):
      self.lines = iter([
        json.dumps({"type": "response.output_text.delta", "delta": "Hello "}) + "\n",
        json.dumps({"type": "tool_call", "tool": "read"}) + "\n",
        json.dumps({"type": "response.output_text.delta", "delta": "world"}) + "\n",
        json.dumps({"type": "turn.complete", "final_text": "Hello world"}) + "\n",
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

  provider = PiAgentProvider(model="gemini-flash-latest")
  events = [event async for event in provider.stream(_request(tmp_path))]

  assert any(isinstance(event, AgentRawProviderEvent) for event in events)
  assert [event.content for event in events if isinstance(event, AgentTextDelta)] == ["Hello ", "world"]
  assert isinstance(events[-1], AgentComplete)
  assert events[-1].result.text == "Hello world"


@pytest.mark.asyncio
async def test_stream_raises_on_nonzero_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  class FakeStdout:
    def readline(self):
      return ""

    def close(self):
      return None

  class FakeStdin:
    def write(self, text):
      return None

    def close(self):
      return None

  class FakeStderr:
    def read(self):
      return "stream failed"

    def close(self):
      return None

  class FakeProcess:
    def __init__(self, *args, **kwargs):
      self.stdin = FakeStdin()
      self.stdout = FakeStdout()
      self.stderr = FakeStderr()
      self.returncode = 1

    def wait(self):
      return 1

    def terminate(self):
      return None

  monkeypatch.setattr(subprocess, "Popen", FakeProcess)

  provider = PiAgentProvider(model="gemini-flash-latest")
  with pytest.raises(ProviderFailure, match="stream failed"):
    [event async for event in provider.stream(_request(tmp_path))]


def test_malformed_json_line_becomes_raw_event(tmp_path: Path):
  provider = PiAgentProvider(model="gemini-flash-latest")
  event = provider._raw_event_from_line("not json")

  assert event.event_type == "malformed_json"
  assert event.payload["line"] == "not json"
