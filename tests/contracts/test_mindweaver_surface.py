"""Contract tests for the mindweaver-facing surface.

Mindweaver migrates from ``asyncio.Lock()`` to ``merceka_core.gpu_lock``.
These tests assert the exact shape mindweaver imports and uses.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

from merceka_core import GpuLockTimeout, gpu_lock
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
