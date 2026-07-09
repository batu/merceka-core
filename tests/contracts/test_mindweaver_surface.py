"""Contract tests for the mindweaver-facing surface.

Mindweaver migrates from ``asyncio.Lock()`` to ``merceka_core.gpu_lock``.
These tests assert the exact shape mindweaver imports and uses.
"""
from __future__ import annotations

import errno
import fcntl
import inspect
import time

import pytest

from merceka_core import (
  Agent,
  AgentComplete,
  AgentProfile,
  AgentRawProviderEvent,
  AgentRequest,
  AgentResult,
  AgentTextDelta,
  GpuLockTimeout,
  ProviderFailure,
  RawProviderEvent,
  gpu_lock,
)
from merceka_core.agents import ClaudeCodeAgentProvider, CodexAgentProvider
from merceka_core.llm import LLM


def test_gpu_lock_is_factory_not_instance():
  """``gpu_lock`` must be callable — mindweaver does ``async with gpu_lock():``."""
  assert callable(gpu_lock)


def test_gpu_lock_accepts_timeout_kwarg():
  """Mindweaver may pass ``timeout=...`` to bound wait time."""
  sig = inspect.signature(gpu_lock)
  assert "timeout" in sig.parameters
  assert sig.parameters["timeout"].default is None


@pytest.mark.asyncio
async def test_gpu_lock_basic_exclusion_in_single_process():
  """Two sequential acquisitions in the same process succeed — the lock
  is re-entrant across separate acquire/release cycles but NOT within a
  single holder (standard flock semantics)."""
  async with gpu_lock(timeout=2):
    pass
  async with gpu_lock(timeout=2):
    pass


@pytest.mark.asyncio
async def test_gpu_lock_timeout_raises():
  """Acquiring while already held in the SAME fd does NOT timeout
  (flock on the same fd is re-entrant), but acquiring via a separate fd
  while held does. We simulate this by holding one CM and trying a
  second with a short timeout from the same event loop — which will
  timeout because we use a separate fd via os.open each call."""
  async with gpu_lock():
    with pytest.raises(GpuLockTimeout):
      # Second acquisition opens a fresh fd → blocked, we time out fast.
      async with gpu_lock(timeout=0.2):
        pytest.fail("Should not have acquired while held")


@pytest.mark.asyncio
async def test_acquire_times_out_when_loop_stalls_past_deadline(monkeypatch):
  """Regression: a stall past the deadline must not let a late release be
  acquired after the timeout window.

  Simulate the race directly against ``_acquire_with_deadline``: the first
  non-blocking ``flock`` attempt reports "held" (would-block), then the
  ``asyncio.sleep`` stalls real (monotonic) time past the deadline while the
  holder "releases" — so a second ``flock`` attempt *would* succeed. The
  contract is that we time out (return ``False``) rather than acquire late,
  which means ``flock`` must be attempted exactly once."""
  from merceka_core.resources import gpu

  attempts = {"n": 0}

  def fake_flock(fd, op):
    if op & fcntl.LOCK_NB:
      attempts["n"] += 1
      if attempts["n"] == 1:
        raise OSError(errno.EAGAIN, "held by someone else")
      # Any later attempt would succeed — the holder released mid-stall.
      return None
    return None

  async def stalling_sleep(_delay):
    # Block real monotonic time (which loop.time() reads) well past the
    # 0.01s deadline, mimicking a stalled event loop.
    time.sleep(0.05)

  monkeypatch.setattr(gpu.fcntl, "flock", fake_flock)
  monkeypatch.setattr(gpu.asyncio, "sleep", stalling_sleep)

  acquired = await gpu._acquire_with_deadline(fd=-1, timeout=0.01)

  assert acquired is False, "acquired the lock after the deadline (late success)"
  assert attempts["n"] == 1, (
    f"flock retried after the deadline had passed (attempts={attempts['n']})"
  )


def test_agenerate_with_resource_exists():
  """Mindweaver calls this method in youtube.explain_screenshot."""
  llm = LLM("openrouter/anthropic/claude-sonnet-4-5")
  assert inspect.iscoroutinefunction(llm.agenerate_with_resource)
  sig = inspect.signature(llm.agenerate_with_resource)
  params = list(sig.parameters)
  assert params[0] == "message"
  assert params[1] == "resource_path"


def test_openrouter_claude_does_not_raise_at_init():
  """Regression guard — the old guard at generate_with_resource (:450)
  raised ValueError for Claude CLI models. OpenRouter-routed Claude
  must work."""
  # Just constructing should not raise.
  llm = LLM("openrouter/anthropic/claude-sonnet-4-5")
  assert llm.model_name == "openrouter/anthropic/claude-sonnet-4-5"


def test_agent_surface_exports_mindweaver_imports():
  assert Agent is not None
  assert AgentProfile.READ_ONLY == "read_only"
  assert AgentRequest is not None
  assert AgentResult is not None
  assert AgentTextDelta is not None
  assert AgentRawProviderEvent is not None
  assert AgentComplete is not None
  assert RawProviderEvent is not None
  assert ProviderFailure is not None


def test_claude_code_agent_provider_is_public_for_mindweaver():
  provider = ClaudeCodeAgentProvider(model="sonnet")

  assert provider.model == "sonnet"


def test_codex_agent_provider_is_public_for_mindweaver():
  provider = CodexAgentProvider(model="gpt-5.5-high")

  assert provider.model == "gpt-5.5-high"
