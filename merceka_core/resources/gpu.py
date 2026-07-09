"""Cross-process GPU serialization via an advisory file lock.

Ollama + WhisperX on the same RTX 4090 OOM together. Multiple processes
(slab doing vision triage, mindweaver enriching a transcript, a CLI
doing ad-hoc work) must serialize their GPU usage even though they live
in different Python interpreters.

Primitive: ``fcntl.flock(fd, LOCK_EX)`` on
``~/.local/state/utolye/gpu.lock``.

Why this directory? ``$XDG_STATE_HOME`` (default
``~/.local/state``) is the XDG-spec location for persistent state. We
cannot use ``~/.cache/`` — on some distros that path is mounted as
tmpfs and can disappear mid-session, which would orphan the lock file.

Why ``flock`` specifically? It's advisory, held on the open file
descriptor, and released by the kernel when the process dies —
including on ``SIGKILL``. No stale-lock cleanup code needed. The lock
file itself stays on disk; it's the fd state the kernel tracks.

Usage::

    from merceka_core import gpu_lock

    async def transcribe(audio):
        async with gpu_lock(timeout=600):
            return await whisperx.transcribe(audio)
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import fcntl
import os
from pathlib import Path
from typing import AsyncIterator

from merceka_core.errors import GpuLockTimeout

GPU_LOCK_PATH: Path = Path.home() / ".local" / "state" / "utolye" / "gpu.lock"

# How often to retry a non-blocking acquisition while waiting. Small
# enough that a freed lock is picked up promptly and the timeout is
# honored within a tight bound; large enough that the poll is cheap.
_POLL_INTERVAL: float = 0.01

# flock reports "held by someone else" via one of these errnos depending
# on the platform.
_WOULD_BLOCK = {errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK}


async def _acquire_with_deadline(fd: int, timeout: float | None) -> bool:
  """Poll ``LOCK_EX | LOCK_NB`` until acquired or the deadline passes.

  Returns ``True`` if the lock was acquired, ``False`` on timeout. This
  runs entirely on the event loop: each ``flock`` attempt is
  non-blocking and returns immediately, and we ``await asyncio.sleep``
  between attempts so cancellation (including :class:`asyncio.TimeoutError`
  from an outer ``wait_for``) unwinds cleanly with no orphaned executor
  thread. The caller owns the fd, so all cleanup happens in its
  ``finally``.
  """
  loop = asyncio.get_running_loop()
  deadline = None if timeout is None else loop.time() + timeout
  while True:
    try:
      fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
      return True
    except OSError as exc:
      if exc.errno not in _WOULD_BLOCK:
        raise
    if deadline is not None and loop.time() >= deadline:
      return False
    delay = _POLL_INTERVAL
    if deadline is not None:
      delay = min(delay, deadline - loop.time())
    await asyncio.sleep(max(0.0, delay))
    # The event loop may have stalled well past the deadline while we
    # were sleeping (or between wakeups). Re-check before the next
    # attempt: without this, a holder that releases during the stall
    # would let us acquire *after* the caller's timeout window, so a
    # late success masquerades as an on-time one. Only the first attempt
    # (top of the loop, before any sleep) is exempt.
    if deadline is not None and loop.time() >= deadline:
      return False


@contextlib.asynccontextmanager
async def gpu_lock(timeout: float | None = None) -> AsyncIterator[None]:
  """Async context manager that serializes GPU access across processes.

  Args:
    timeout: Seconds to wait for acquisition. ``None`` waits
      indefinitely. Raises :class:`~merceka_core.errors.GpuLockTimeout`
      on timeout.

  Notes:
    Acquisition uses non-blocking ``flock(LOCK_NB)`` polling against a
    monotonic deadline, so it never blocks the event loop and is
    cancellation-safe: on timeout or cancellation no work is left
    running in a background thread. The kernel releases the lock when
    the fd is closed — including on unexpected process death — so we do
    NOT need a stale-lock cleanup pass.
  """
  GPU_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
  fd = os.open(GPU_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o600)
  acquired = False
  try:
    acquired = await _acquire_with_deadline(fd, timeout)
    if not acquired:
      raise GpuLockTimeout(f"gpu_lock({GPU_LOCK_PATH}) timed out after {timeout}s")
    yield
  finally:
    if acquired:
      try:
        fcntl.flock(fd, fcntl.LOCK_UN)
      except OSError:
        # fd may have been closed by the kernel on process shutdown.
        pass
    os.close(fd)
