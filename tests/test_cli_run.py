"""_cli.run_cli: provider CLI calls stop their whole process group on timeout.

codex and pi are node wrappers around the real worker, so a timeout that kills
only the direct child leaves the worker running after the caller gave up.
"""

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from merceka_core import _cli


@pytest.fixture(autouse=True)
def _short_grace(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(_cli, "_TIMEOUT_GRACE_S", 0.5)


def _script(tmp_path: Path, body: str) -> str:
  path = tmp_path / "fake_cli"
  path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
  path.chmod(0o755)
  return str(path)


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


def _wrapper_with_grandchild(pid_file: Path) -> str:
  """A CLI that starts a long-running worker and, like a thin wrapper that
  exits on SIGTERM, passes no signal on to it."""
  return f"""
    import subprocess, sys, time
    sys.stdin.read()
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    open({str(pid_file)!r}, "w").write(str(worker.pid))
    time.sleep(60)
  """


def test_timeout_stops_the_whole_process_group(tmp_path):
  pid_file = tmp_path / "worker.pid"
  cmd = [_script(tmp_path, _wrapper_with_grandchild(pid_file))]
  started = time.monotonic()

  with pytest.raises(subprocess.TimeoutExpired):
    _cli.run_cli(cmd, input="prompt", timeout=1)

  assert time.monotonic() - started < 5
  worker = int(pid_file.read_text())
  try:
    assert _dies_within(worker, 3), "the wrapper's worker outlived the timeout"
  finally:
    if _alive(worker):
      os.kill(worker, 9)


def test_returns_exit_status_and_both_streams(tmp_path):
  cmd = [_script(tmp_path, """
    import sys
    text = sys.stdin.read()
    print("out:" + text)
    print("err", file=sys.stderr)
    sys.exit(3)
  """)]

  result = _cli.run_cli(cmd, input="hello", timeout=10)

  assert result.returncode == 3
  assert result.stdout == "out:hello\n"
  assert result.stderr == "err\n"
  assert result.args == cmd


def test_env_and_cwd_reach_the_child(tmp_path):
  cmd = [_script(tmp_path, """
    import os
    print(os.getcwd(), os.environ.get("RUN_CLI_PROBE"))
  """)]
  workdir = tmp_path / "work"
  workdir.mkdir()

  result = _cli.run_cli(cmd, env={"RUN_CLI_PROBE": "seen"}, cwd=str(workdir))

  assert result.stdout.split() == [str(workdir.resolve()), "seen"]

