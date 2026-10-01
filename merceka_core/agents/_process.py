"""Process plumbing shared by the CLI agent providers (claude, codex, pi).

Each provider keeps its own command line, prompt and event parsing. The
lifecycle code that is identical across them lives here.

Every CLI runs in its own session, so it leads a new process group. Stopping
a run signals that whole group: ``codex`` is a node wrapper around a native
binary, and killing only the wrapper would leave the agent running (and, under
a WRITE profile, editing files) after the caller gave up.
"""

from __future__ import annotations

import asyncio
import json
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from merceka_core._cli import signal_process_group
from merceka_core.agent import ProviderFailure, RawProviderEvent

# How long a CLI gets to exit after SIGTERM before its process group is killed.
TERMINATE_GRACE_SECONDS = 5.0
# How long to wait for stderr to reach EOF once the CLI has exited. A descendant
# can keep the pipe open; the error text is diagnostic, so it is not worth a hang.
STDERR_GRACE_SECONDS = 1.0

_T = TypeVar("_T")


def raw_event_from_line(line: str, provider: str) -> RawProviderEvent:
  """One JSON line of CLI output as a raw event; unparseable lines are kept as such."""
  try:
    payload: Any = json.loads(line)
  except json.JSONDecodeError as exc:
    return RawProviderEvent(
      provider=provider,
      event_type="malformed_json",
      payload={"line": line, "error": str(exc)},
    )
  event_type = str(payload.get("type", "raw")) if isinstance(payload, dict) else "raw"
  return RawProviderEvent(provider=provider, event_type=event_type, payload=payload)


def raw_events_from_stdout(stdout: str, provider: str) -> list[RawProviderEvent]:
  return [raw_event_from_line(line, provider) for line in stdout.splitlines() if line.strip()]


def start(cmd: list[str], *, cwd: str, env: dict[str, str]) -> subprocess.Popen[str]:
  """Start a CLI with piped stdio as the leader of a new process group."""
  return subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
    cwd=cwd,
    env=env,
    start_new_session=True,
  )


async def run(
  cmd: list[str],
  *,
  input: str,
  timeout: float,
  cwd: str,
  env: dict[str, str],
  label: str,
) -> subprocess.CompletedProcess[str]:
  """Run a CLI to completion, feeding ``input`` on stdin.

  On timeout the process group is stopped and ProviderFailure is raised. On
  cancellation the process group is stopped before CancelledError propagates.
  """
  process = start(cmd, cwd=cwd, env=env)
  try:
    stdout, stderr = await asyncio.wait_for(
      asyncio.to_thread(process.communicate, input), timeout
    )
  except TimeoutError:
    await asyncio.to_thread(terminate_process, process)
    raise ProviderFailure(f"{label} timed out after {timeout:g}s") from None
  except BaseException:
    await asyncio.shield(asyncio.to_thread(terminate_process, process))
    raise
  return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)


class Stream:
  """A CLI whose stdout is read line by line while it runs.

  stderr is drained on a thread from the start, so a chatty CLI never blocks on
  a full pipe. Every wait (sending the prompt, each line, the exit) counts
  against one deadline of ``timeout`` seconds; running past it raises
  ProviderFailure. Blocking calls run in threads, off the event loop. The
  prompt is written inside the caller's cleanup scope, so a CLI that dies at
  once is still torn down. Always ``await close()`` when done.
  """

  def __init__(
    self, cmd: list[str], *, cwd: str, env: dict[str, str], timeout: float, label: str
  ) -> None:
    self.label = label
    self.timeout = timeout
    self._deadline = time.monotonic() + timeout
    self._prompt_delivered = False
    self.process = start(cmd, cwd=cwd, env=env)
    if self.process.stdin is None or self.process.stdout is None or self.process.stderr is None:
      terminate_process(self.process)
      raise ProviderFailure(f"{label} did not expose stdio pipes")
    self._stdin = self.process.stdin
    self._stdout = self.process.stdout
    self._stderr = _Drain(self.process.stderr)

  async def send(self, text: str) -> None:
    """Write the prompt and close stdin.

    A CLI that exits before reading it is not reported here: its exit status
    and stderr, read by ``finish()``, say why.
    """
    self._prompt_delivered = await self._bounded(_write_and_close, self._stdin, text)

  async def readline(self) -> str:
    """The next line of stdout, or "" at EOF."""
    return await self._bounded(self._stdout.readline)

  async def finish(self) -> tuple[int, str]:
    """Wait for the CLI to exit; return its exit status and stderr."""
    returncode = await self._bounded(self.process.wait)
    stderr = await asyncio.to_thread(self._stderr.text, STDERR_GRACE_SECONDS)
    if returncode == 0 and not self._prompt_delivered:
      detail = stderr.strip() or "no error output"
      raise ProviderFailure(f"{self.label} exited before reading its prompt: {detail}")
    return returncode, stderr

  async def close(self) -> None:
    """Stop the process group if the CLI is still running, and close the pipes."""
    await asyncio.to_thread(self._close)

  def _close(self) -> None:
    if self.process.returncode is None:
      terminate_process(self.process)
    close_pipe(self._stdin)
    close_pipe(self._stdout)
    # A pipe cannot be closed while the drain thread is blocked reading it.
    if self._stderr.finished(STDERR_GRACE_SECONDS):
      close_pipe(self.process.stderr)

  async def _bounded(self, fn: Callable[..., _T], *args: Any) -> _T:
    remaining = self._deadline - time.monotonic()
    if remaining <= 0:
      raise ProviderFailure(f"{self.label} timed out after {self.timeout:g}s")
    try:
      return await asyncio.wait_for(asyncio.to_thread(fn, *args), remaining)
    except TimeoutError:
      raise ProviderFailure(f"{self.label} timed out after {self.timeout:g}s") from None


class _Drain:
  """Reads a pipe to EOF on a daemon thread."""

  def __init__(self, pipe: Any) -> None:
    self._text = ""
    self._thread = threading.Thread(target=self._read, args=(pipe,), daemon=True)
    self._thread.start()

  def _read(self, pipe: Any) -> None:
    try:
      self._text = pipe.read()
    except (OSError, ValueError):
      pass

  def finished(self, timeout: float) -> bool:
    self._thread.join(timeout)
    return not self._thread.is_alive()

  def text(self, timeout: float) -> str:
    self._thread.join(timeout)
    return self._text


def _write_and_close(pipe: Any, text: str) -> bool:
  """Write ``text`` and close the pipe; False when the reader was already gone."""
  try:
    pipe.write(text)
  except BrokenPipeError:
    delivered = False
  else:
    delivered = True
  try:
    pipe.close()  # Closes the descriptor even when the final flush fails.
  except BrokenPipeError:
    delivered = False
  return delivered


def terminate_process(process: subprocess.Popen[str]) -> None:
  """Stop the process group: SIGTERM, then SIGKILL once the grace period is over.

  Blocks for up to the grace period, so async callers run it in a thread.
  """
  if process.poll() is None:
    signal_process_group(process, signal.SIGTERM)
    try:
      process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
      pass
  # SIGKILL whatever is left: the leader if it ignored SIGTERM, and any
  # descendant still in the group after the leader exited.
  signal_process_group(process, signal.SIGKILL)
  process.wait()



def close_pipe(pipe: Any) -> None:
  if pipe is not None:
    pipe.close()
