"""Contract for the synchronous GPU lock.

``gpu_lock_sync`` exists for callers with no event loop — batch renderers,
CLIs, worker scripts. The property that matters is not that it works alone
but that it contends with the async form: both take ``LOCK_EX`` on the same
file, so a sync holder must block an async one and vice versa. If that
mutual exclusion breaks, a long render and a WhisperX job will happily share
the GPU and OOM it.
"""

from __future__ import annotations

import asyncio
import inspect
import subprocess
import sys
import time

import pytest

from merceka_core import gpu_lock, gpu_lock_sync
from merceka_core.errors import GpuLockTimeout


@pytest.fixture(autouse=True)
def _isolate_gpu_lock(tmp_path, monkeypatch):
  """Point both locks at a per-test file so the live GPU lock is untouched."""
  monkeypatch.setattr("merceka_core.resources.gpu.GPU_LOCK_PATH", tmp_path / "gpu.lock")


def test_gpu_lock_sync_signature_matches_async():
  """Callers should be able to swap one for the other without changing kwargs."""
  sync_sig = inspect.signature(gpu_lock_sync)
  async_sig = inspect.signature(gpu_lock)
  assert list(sync_sig.parameters) == list(async_sig.parameters)
  assert sync_sig.parameters["timeout"].default is None


def test_gpu_lock_sync_releases_between_acquisitions():
  """Sequential acquisitions succeed, so a per-chunk loop does not self-deadlock."""
  for _ in range(3):
    with gpu_lock_sync(timeout=2):
      pass


def test_gpu_lock_sync_releases_on_exception():
  """An error inside the body must not leave the lock held."""
  with pytest.raises(RuntimeError):
    with gpu_lock_sync(timeout=2):
      raise RuntimeError("render failed")
  with gpu_lock_sync(timeout=2):
    pass


@pytest.mark.asyncio
async def test_sync_holder_blocks_async_waiter(tmp_path):
  """The whole point: a sync holder in another process excludes an async waiter."""
  lock_path = tmp_path / "gpu.lock"
  holder = subprocess.Popen(
    [
      sys.executable,
      "-c",
      "import sys, time\n"
      "from pathlib import Path\n"
      "import merceka_core.resources.gpu as g\n"
      "g.GPU_LOCK_PATH = Path(sys.argv[1])\n"
      "from merceka_core import gpu_lock_sync\n"
      "with gpu_lock_sync(timeout=10):\n"
      "    Path(sys.argv[2]).write_text('held')\n"
      "    time.sleep(float(sys.argv[3]))\n",
      str(lock_path),
      str(tmp_path / "ready"),
      "1.5",
    ]
  )
  try:
    ready = tmp_path / "ready"
    deadline = time.monotonic() + 10
    while not ready.exists() and time.monotonic() < deadline:
      await asyncio.sleep(0.02)
    assert ready.exists(), "sync holder never acquired the lock"

    with pytest.raises(GpuLockTimeout):
      async with gpu_lock(timeout=0.3):
        pass
  finally:
    holder.wait(timeout=15)

  # Once the sync holder exits the async form acquires immediately.
  async with gpu_lock(timeout=5):
    pass


def test_async_holder_blocks_sync_waiter(tmp_path):
  """And the reverse direction, so neither flavor can starve unnoticed."""
  lock_path = tmp_path / "gpu.lock"
  holder = subprocess.Popen(
    [
      sys.executable,
      "-c",
      "import asyncio, sys\n"
      "from pathlib import Path\n"
      "import merceka_core.resources.gpu as g\n"
      "g.GPU_LOCK_PATH = Path(sys.argv[1])\n"
      "from merceka_core import gpu_lock\n"
      "async def main():\n"
      "    async with gpu_lock(timeout=10):\n"
      "        Path(sys.argv[2]).write_text('held')\n"
      "        await asyncio.sleep(float(sys.argv[3]))\n"
      "asyncio.run(main())\n",
      str(lock_path),
      str(tmp_path / "ready"),
      "1.5",
    ]
  )
  try:
    ready = tmp_path / "ready"
    deadline = time.monotonic() + 10
    while not ready.exists() and time.monotonic() < deadline:
      time.sleep(0.02)
    assert ready.exists(), "async holder never acquired the lock"

    with pytest.raises(GpuLockTimeout):
      with gpu_lock_sync(timeout=0.3):
        pass
  finally:
    holder.wait(timeout=15)

  with gpu_lock_sync(timeout=5):
    pass


def test_gpu_lock_sync_survives_holder_death(tmp_path):
  """flock is kernel-released, so a killed render must not wedge the lock."""
  lock_path = tmp_path / "gpu.lock"
  holder = subprocess.Popen(
    [
      sys.executable,
      "-c",
      "import sys, time\n"
      "from pathlib import Path\n"
      "import merceka_core.resources.gpu as g\n"
      "g.GPU_LOCK_PATH = Path(sys.argv[1])\n"
      "from merceka_core import gpu_lock_sync\n"
      "with gpu_lock_sync(timeout=10):\n"
      "    Path(sys.argv[2]).write_text('held')\n"
      "    time.sleep(60)\n",
      str(lock_path),
      str(tmp_path / "ready"),
    ]
  )
  ready = tmp_path / "ready"
  deadline = time.monotonic() + 10
  while not ready.exists() and time.monotonic() < deadline:
    time.sleep(0.02)
  assert ready.exists(), "holder never acquired the lock"

  holder.kill()
  holder.wait(timeout=15)
  with gpu_lock_sync(timeout=5):
    pass
