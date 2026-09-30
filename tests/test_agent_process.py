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
  provider = PROVIDERS[provider_name](fake_cli(_wrapper_with_grandchild(pid_file)), 1)
  started = time.monotonic()

  with pytest.raises(ProviderFailure, match="timed out after 1s"):
    await provider.run(_request(tmp_path))

  grandchild = _read_pid(pid_file)
  try:
    assert time.monotonic() - started < 5
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
