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
import os
import signal
import subprocess
from typing import Any

from merceka_core.agent import ProviderFailure, RawProviderEvent

# How long a CLI gets to exit after SIGTERM before its process group is killed.
TERMINATE_GRACE_SECONDS = 5.0


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


def terminate_process(process: subprocess.Popen[str]) -> None:
  """Stop the process group: SIGTERM, then SIGKILL once the grace period is over.

  Blocks for up to the grace period, so async callers run it in a thread.
  """
  if process.poll() is None:
    _signal_group(process, signal.SIGTERM)
    try:
      process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
      pass
  # SIGKILL whatever is left: the leader if it ignored SIGTERM, and any
  # descendant still in the group after the leader exited.
  _signal_group(process, signal.SIGKILL)
  process.wait()


def _signal_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
  try:
    os.killpg(process.pid, sig)
  except ProcessLookupError:
    pass  # The group is gone.
  except PermissionError:
    # macOS refuses to signal a group whose only member is an unreaped zombie.
    if process.poll() is None:
      process.send_signal(sig)


def close_pipe(pipe: Any) -> None:
  if pipe is not None:
    pipe.close()
