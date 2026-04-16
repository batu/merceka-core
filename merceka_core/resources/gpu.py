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
import fcntl
import os
from pathlib import Path
from typing import AsyncIterator

from merceka_core.errors import GpuLockTimeout

GPU_LOCK_PATH: Path = Path.home() / ".local" / "state" / "utolye" / "gpu.lock"


def _acquire_blocking(fd: int) -> None:
  """Blocking LOCK_EX acquisition — run this in an executor."""
  fcntl.flock(fd, fcntl.LOCK_EX)


@contextlib.asynccontextmanager
async def gpu_lock(timeout: float | None = None) -> AsyncIterator[None]:
  """Async context manager that serializes GPU access across processes.

  Args:
    timeout: Seconds to wait for acquisition. ``None`` waits
      indefinitely. Raises :class:`~merceka_core.errors.GpuLockTimeout`
      on timeout.

  Notes:
    The underlying ``fcntl.flock`` call is blocking; we dispatch it to
    the default thread executor so the event loop remains responsive.
    The kernel releases the lock when the fd is closed — including on
    unexpected process death — so we do NOT need a stale-lock cleanup
    pass.
  """
  GPU_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
  fd = os.open(GPU_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o600)
  acquired = False
  try:
    loop = asyncio.get_running_loop()
    acquire_future = loop.run_in_executor(None, _acquire_blocking, fd)
    try:
      if timeout is None:
        await acquire_future
      else:
        await asyncio.wait_for(acquire_future, timeout=timeout)
      acquired = True
    except asyncio.TimeoutError as exc:
      # The executor task is still running and will eventually acquire.
      # That's unavoidable with blocking flock; the caller treats this
      # as "give up", and the eventual acquisition is followed
      # immediately by fd close, which releases the lock.
      raise GpuLockTimeout(
        f"gpu_lock({GPU_LOCK_PATH}) timed out after {timeout}s"
      ) from exc
    yield
  finally:
    if acquired:
      try:
        fcntl.flock(fd, fcntl.LOCK_UN)
      except OSError:
        # fd may have been closed by the kernel on process shutdown.
        pass
    os.close(fd)
