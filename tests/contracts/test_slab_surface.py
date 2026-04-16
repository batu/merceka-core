"""Contract tests — slab depends on these exact names and shapes.

If any of these assertions fail, the slab pipeline (videototext, slab
hypothesize/author) will break at import time or first call. Treat a
red test here as blocking.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from merceka_core import (
  GPU_LOCK_PATH,
  GpuLockTimeout,
  VideoBackendError,
  VideoNotFoundError,
  VideoUploadError,
  gpu_lock,
)
from merceka_core.llm import LLM


def test_exception_hierarchy():
  """The four exception classes exist and inherit sensibly."""
  assert issubclass(VideoUploadError, Exception)
  assert issubclass(VideoBackendError, Exception)
  assert issubclass(VideoNotFoundError, FileNotFoundError)
  assert issubclass(GpuLockTimeout, TimeoutError)
  # VideoNotFoundError being a FileNotFoundError is load-bearing: the
  # LLM.generate fallback cascade already catches FileNotFoundError.
  assert issubclass(VideoNotFoundError, FileNotFoundError)


def test_gpu_lock_path():
  """gpu_lock points at the XDG persistent state dir, not tmpfs."""
  # ~/.cache/ is tmpfs on some distros — must NOT be in the lock path.
  assert ".cache" not in str(GPU_LOCK_PATH)
  assert str(GPU_LOCK_PATH).endswith(".local/state/utolye/gpu.lock")


def test_gpu_lock_is_async_context_manager():
  """gpu_lock(timeout=...) returns an async context manager."""
  cm = gpu_lock(timeout=5)
  # Async CM exposes __aenter__/__aexit__.
  assert hasattr(cm, "__aenter__")
  assert hasattr(cm, "__aexit__")


def test_generate_with_resource_signature_openrouter_claude():
  """OpenRouter-Claude LLM routes through the cloud path (use_openrouter=True)."""
  llm = LLM("openrouter/anthropic/claude-sonnet-4-5")
  assert llm.use_openrouter is True
  assert llm.use_claude is False
  assert llm.use_gemini is False
  # Method exists, takes (message, resource_path).
  sig = inspect.signature(llm.generate_with_resource)
  params = list(sig.parameters)
  assert params[0] == "message"
  assert params[1] == "resource_path"


def test_generate_with_video_signature():
  """generate_with_video exists on LLM with the documented signature."""
  llm = LLM("gemini/gemini-flash-latest")
  assert llm.use_gemini is True
  sig = inspect.signature(llm.generate_with_video)
  params = sig.parameters
  assert "message" in params
  assert "video_paths" in params
  assert "timeout_s" in params
  # timeout_s defaults to 300 per the card spec.
  assert params["timeout_s"].default == 300.0


def test_generate_with_video_rejects_non_gemini():
  """Calling generate_with_video on a non-Gemini model raises ValueError."""
  llm = LLM("openrouter/anthropic/claude-sonnet-4-5")
  with pytest.raises(ValueError, match="Gemini"):
    llm.generate_with_video("prompt", Path("/tmp/does_not_matter.mp4"))


def test_agenerate_with_video_is_async():
  """agenerate_with_video is a coroutine function."""
  llm = LLM("gemini/gemini-flash-latest")
  assert inspect.iscoroutinefunction(llm.agenerate_with_video)


def test_video_not_found_raised_eagerly(tmp_path):
  """generate_with_video raises VideoNotFoundError before touching the SDK."""
  llm = LLM("gemini/gemini-flash-latest")
  missing = tmp_path / "does_not_exist.mp4"
  with pytest.raises(VideoNotFoundError):
    llm.generate_with_video("prompt", missing)
