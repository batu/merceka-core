"""Process lifecycle of the CLI agent providers, exercised with real fake CLIs.

Each fake is a small Python script run in place of claude, codex or pi. The
tests cover what mocks cannot: process groups, pipes filling up, children that
ignore SIGTERM or exit before reading the prompt.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from merceka_core.agent import AgentProvider, AgentRequest, ProviderFailure
from merceka_core.agents import _process
from merceka_core.agents.claude_code import ClaudeCodeAgentProvider
from merceka_core.agents.codex import CodexAgentProvider
from merceka_core.agents.pi import PiAgentProvider

PROVIDERS: dict[str, Callable[[str, int], AgentProvider]] = {
  "claude": lambda binary, timeout: ClaudeCodeAgentProvider(
    model="sonnet", claude_binary=binary, timeout_seconds=timeout
  ),
  "codex": lambda binary, timeout: CodexAgentProvider(codex_binary=binary, timeout_seconds=timeout),
  "pi": lambda binary, timeout: PiAgentProvider(pi_binary=binary, timeout_seconds=timeout),
}


@pytest.fixture
def fake_cli(tmp_path: Path) -> Callable[[str], str]:
  def write(body: str) -> str:
    path = tmp_path / f"fake_cli_{len(list(tmp_path.glob('fake_cli_*')))}"
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    path.chmod(0o755)
    return str(path)

  return write


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> list[subprocess.Popen]:
  """Every real Popen the providers start, so tests can inspect it afterwards."""
  processes: list[subprocess.Popen] = []
  real_popen = subprocess.Popen

  def spy(*args, **kwargs):
    process = real_popen(*args, **kwargs)
    processes.append(process)
    return process

  monkeypatch.setattr(subprocess, "Popen", spy)
  return processes


@pytest.fixture(autouse=True)
def _short_grace(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(_process, "TERMINATE_GRACE_SECONDS", 0.5)


def _request(root: Path, message: str = "question") -> AgentRequest:
  return AgentRequest(message=message, system_prompt="system", roots=(root,))


def _alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  return True


def _dies_within(pid: int, seconds: float) -> bool:
  deadline = time.monotonic() + seconds
  while time.monotonic() < deadline:
    if not _alive(pid):
      return True
    time.sleep(0.05)
  return False


def _read_pid(pid_file: Path) -> int:
  deadline = time.monotonic() + 5
  while time.monotonic() < deadline:
    if pid_file.exists() and pid_file.read_text():
      return int(pid_file.read_text())
    time.sleep(0.05)
  raise AssertionError("fake CLI never started its grandchild")


def _kill_if_alive(pid: int) -> None:
  if _alive(pid):
    os.kill(pid, signal.SIGKILL)


def _wrapper_with_grandchild(pid_file: Path, *, announce: bool = False) -> str:
  """A CLI that starts a long-running grandchild and, like a thin wrapper that
  exits on SIGTERM, passes no signal on to it."""
  announce_line = 'print(json.dumps({"type": "started"}), flush=True)' if announce else ""
  return f"""
    import json, subprocess, sys, time
    sys.stdin.read()
    grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    open({str(pid_file)!r}, "w").write(str(grandchild.pid))
    {announce_line}
    time.sleep(60)
  """


# --- run() ---


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", sorted(PROVIDERS))
async def test_run_timeout_raises_provider_failure_and_stops_the_process_group(
  provider_name, fake_cli, tmp_path
):
  pid_file = tmp_path / "grandchild.pid"
  provider = PROVIDERS[provider_name](fake_cli(_wrapper_with_grandchild(pid_file)), 2)
  started = time.monotonic()
  task = asyncio.create_task(provider.run(_request(tmp_path)))
  grandchild = await asyncio.to_thread(_read_pid, pid_file)
  try:
    with pytest.raises(ProviderFailure, match="timed out after 2s"):
      await task

    assert time.monotonic() - started < 8
    assert _dies_within(grandchild, 3)
  finally:
    _kill_if_alive(grandchild)


@pytest.mark.asyncio
async def test_cancelled_run_stops_the_process_group(fake_cli, tmp_path):
  pid_file = tmp_path / "grandchild.pid"
  provider = PROVIDERS["codex"](fake_cli(_wrapper_with_grandchild(pid_file)), 60)
  task = asyncio.create_task(provider.run(_request(tmp_path)))
  grandchild = await asyncio.to_thread(_read_pid, pid_file)
  try:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await task

    assert _dies_within(grandchild, 3)
  finally:
    _kill_if_alive(grandchild)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", sorted(PROVIDERS))
async def test_run_starts_the_cli_in_its_own_process_group(
  provider_name, fake_cli, spawned, tmp_path
):
  binary = fake_cli("""
    import os, sys
    sys.stdin.read()
    print("pgid", os.getpgid(0), "pid", os.getpid(), file=sys.stderr)
    sys.exit(1)
  """)

  with pytest.raises(ProviderFailure) as failure:
    await PROVIDERS[provider_name](binary, 30).run(_request(tmp_path))

  _, pgid, _, pid = str(failure.value).rsplit(maxsplit=3)
  assert pgid == pid == str(spawned[0].pid)


# --- tearing a stream down ---


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", sorted(PROVIDERS))
async def test_closing_a_stream_does_not_block_the_event_loop(
  provider_name, fake_cli, spawned, tmp_path, monkeypatch
):
  # A child that ignores SIGTERM: stopping it takes the full grace period plus
  # a SIGKILL. Done on the event loop, that would stall it for the whole grace.
  monkeypatch.setattr(_process, "TERMINATE_GRACE_SECONDS", 1.0)
  binary = fake_cli("""
    import json, signal, sys, time
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    sys.stdin.read()
    print(json.dumps({"type": "started"}), flush=True)
    time.sleep(30)
  """)
  provider = PROVIDERS[provider_name](binary, 60)
  stream = provider.stream(_request(tmp_path))
  await anext(stream)
  gaps: list[float] = []

  async def heartbeat():
    last = time.monotonic()
    while True:
      await asyncio.sleep(0.02)
      now = time.monotonic()
      gaps.append(now - last)
      last = now

  beat = asyncio.create_task(heartbeat())
  await asyncio.sleep(0.05)
  started = time.monotonic()
  await stream.aclose()
  closed_after = time.monotonic() - started
  await asyncio.sleep(0.1)  # let the heartbeat record any stall during the close
  beat.cancel()

  assert max(gaps) < 0.5  # a blocking close stalls for at least the 1 s grace
  assert closed_after < 10  # an unbounded wait lasts the child's 30 s sleep
  assert spawned[0].returncode is not None  # the stubborn child is gone, and reaped


@pytest.mark.asyncio
async def test_closing_a_stream_stops_the_whole_process_group(fake_cli, tmp_path):
  pid_file = tmp_path / "grandchild.pid"
  binary = fake_cli(_wrapper_with_grandchild(pid_file, announce=True))
  stream = PROVIDERS["codex"](binary, 30).stream(_request(tmp_path))
  await anext(stream)
  grandchild = _read_pid(pid_file)
  try:
    await stream.aclose()

    assert _dies_within(grandchild, 3)
  finally:
    _kill_if_alive(grandchild)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", sorted(PROVIDERS))
async def test_child_that_exits_before_reading_the_prompt_is_a_provider_failure(
  provider_name, fake_cli, spawned, tmp_path
):
  # The prompt is larger than a pipe buffer, so writing it fails with EPIPE
  # once the child is gone.
  binary = fake_cli("""
    import sys
    print("not logged in", file=sys.stderr)
    sys.exit(3)
  """)
  provider = PROVIDERS[provider_name](binary, 30)

  with pytest.raises(ProviderFailure, match="not logged in"):
    [event async for event in provider.stream(_request(tmp_path, "x" * 1_000_000))]

  child = spawned[0]
  assert child.returncode == 3
  assert child.stdout is not None and child.stdout.closed
  assert child.stderr is not None and child.stderr.closed


# --- stream deadline and stderr ---


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", sorted(PROVIDERS))
async def test_stream_that_goes_silent_times_out(provider_name, fake_cli, spawned, tmp_path):
  binary = fake_cli("""
    import json, sys, time
    sys.stdin.read()
    print(json.dumps({"type": "started"}), flush=True)
    time.sleep(60)
  """)
  provider = PROVIDERS[provider_name](binary, 1)
  started = time.monotonic()

  with pytest.raises(ProviderFailure, match="timed out after 1s"):
    await asyncio.wait_for(_drain(provider.stream(_request(tmp_path))), 10)

  assert time.monotonic() - started < 5
  assert spawned[0].returncode is not None


@pytest.mark.asyncio
async def test_stream_deadline_covers_the_whole_stream_not_each_line(fake_cli, tmp_path):
  # A line every 0.3 s never trips a per-read timeout; the overall one must fire.
  binary = fake_cli("""
    import json, sys, time
    sys.stdin.read()
    while True:
      print(json.dumps({"type": "tick"}), flush=True)
      time.sleep(0.3)
  """)
  started = time.monotonic()

  with pytest.raises(ProviderFailure, match="timed out after 1s"):
    await asyncio.wait_for(_drain(PROVIDERS["pi"](binary, 1).stream(_request(tmp_path))), 10)

  assert time.monotonic() - started < 5


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", sorted(PROVIDERS))
async def test_stream_survives_a_child_that_floods_stderr(provider_name, fake_cli, tmp_path):
  # 300 KB of stderr before any stdout fills the pipe buffer; unless stderr is
  # read while stdout is, the child blocks on write and the stream never ends.
  binary = fake_cli("""
    import json, sys
    sys.stdin.read()
    sys.stderr.write("x" * 300_000)
    sys.stderr.flush()
    print(json.dumps({"type": "result", "subtype": "success"}), flush=True)
  """)
  provider = PROVIDERS[provider_name](binary, 30)

  events = await asyncio.wait_for(_drain(provider.stream(_request(tmp_path))), 10)

  assert events[-1].type == "complete"


@pytest.mark.asyncio
async def test_stream_failure_reports_the_drained_stderr(fake_cli, tmp_path):
  binary = fake_cli("""
    import sys
    sys.stdin.read()
    sys.stderr.write("x" * 300_000 + "the real error")
    sys.exit(2)
  """)

  with pytest.raises(ProviderFailure, match="the real error"):
    await asyncio.wait_for(_drain(PROVIDERS["codex"](binary, 30).stream(_request(tmp_path))), 10)


async def _drain(stream) -> list:
  return [event async for event in stream]
