"""End-to-end local Ollama vision via the plain-name ``LLM("gemma4:...")``
entrypoint, serialized behind the shared ``gpu_lock``.

This test proves two things at once:

1. Merceka already exposes gemma4 as a first-class vision backend — any
   ``LLM(model_name)`` where ``model_name`` does not start with
   ``claude/`` / ``gemini/`` and does not contain ``openrouter`` routes
   through ``_local_call`` → ``ollama_chat`` with the ollama-native
   ``images=[...]`` payload built by ``create_ollama_vision_message``.
   This is the path mindweaver's ``explain_screenshot`` already uses.

2. ``gpu_lock()`` from ``merceka_core.resources`` serializes the call
   with other GPU consumers (WhisperX, slab vision triage). The Phase-0
   card's acceptance bar says "Ollama + WhisperX OOM together on the
   4090 — the lock serializes them" — this test exercises the lock
   around a real ollama call.

Requires a local Ollama daemon at ``http://localhost:11434`` and a
``gemma4:26b`` tag installed (per project memory § Mindweaver Local
Models — that's the benchmark-winning variant mindweaver uses as
``vision_fallback_model``). Runs only with ``-m integration`` and is
additionally gated on ``gemma4:26b`` being present, so CI without a GPU
skips cleanly instead of exploding.

To execute (requires ~19 GB VRAM / ~17 GB for gemma4:26b)::

    uv run pytest tests/integration/test_gemma_local.py -m integration -s
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from merceka_core import gpu_lock
from merceka_core.llm import LLM, list_local_models
from merceka_core.resources.gpu import GPU_LOCK_PATH

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

GEMMA_MODEL = "gemma4:26b"  # benchmark-winning vision variant per MEMORY.md
VIDEO_FIXTURE = Path(
  "/home/batu/Desktop/utolye/videotoframes/videos/find_the_dog.mp4"
)


def _require_ollama_model(model: str) -> None:
  try:
    installed = list_local_models()
  except Exception as exc:  # pragma: no cover — ollama daemon down
    pytest.skip(f"ollama daemon unreachable: {exc}")
  if model not in installed:
    pytest.skip(f"ollama model {model!r} not installed (have: {installed[:5]}...)")


def _require_video_fixture() -> None:
  if not VIDEO_FIXTURE.exists():
    pytest.skip(f"video fixture missing at {VIDEO_FIXTURE}")


def _extract_frame(video: Path, tmp_path: Path) -> Path:
  """Pull a single PNG frame from the video via ffmpeg. Fail fast on error."""
  frame_path = tmp_path / "frame.png"
  subprocess.run(
    [
      "ffmpeg", "-y", "-loglevel", "error",
      "-ss", "1.0",       # 1s into the clip — past any fade-in.
      "-i", str(video),
      "-vframes", "1",    # single frame.
      "-vf", "scale=512:-1",  # downscale for fast vision inference.
      str(frame_path),
    ],
    check=True,
    capture_output=True,
  )
  assert frame_path.exists() and frame_path.stat().st_size > 0
  return frame_path


class TestGemmaLocalVision:
  """``LLM("gemma4:26b")`` + gpu_lock round-trip against a real frame."""

  @pytest.mark.asyncio
  async def test_agenerate_with_resource_gemma4_behind_gpu_lock(self, tmp_path):
    """gemma4 describes a find_the_dog frame. Call is serialized by gpu_lock."""
    _require_ollama_model(GEMMA_MODEL)
    _require_video_fixture()
    frame = _extract_frame(VIDEO_FIXTURE, tmp_path)

    llm = LLM(
      GEMMA_MODEL,
      system_prompt="Describe the screenshot concisely.",
      think=False,  # Thinking hurts gemma4 per enrichment benchmark.
    )

    # Prove the LLM routes through the local Ollama backend — not the
    # cloud / Claude / Gemini paths. If this flips, the test is measuring
    # the wrong thing.
    assert llm.use_openrouter is False
    assert llm.use_claude is False
    assert llm.use_gemini is False

    async with gpu_lock(timeout=120):
      reply = await llm.agenerate_with_resource(
        "What is happening in this image? One sentence.",
        frame,
      )

    assert isinstance(reply, str)
    assert reply.strip(), "gemma4 returned an empty string"

  @pytest.mark.asyncio
  async def test_gpu_lock_is_actually_held_during_call(self, tmp_path):
    """Proof-of-serialization: while one coroutine holds the lock and is
    running inference, a competing ``gpu_lock(timeout=0.1)`` must time out
    with GpuLockTimeout — not slip past the lock.

    This is the failure mode the Phase-0 card cares about: if the lock is
    silently a no-op (e.g. path typo, or gpu_lock stub), two processes
    would run ollama concurrently and OOM the 4090.
    """
    _require_ollama_model(GEMMA_MODEL)
    _require_video_fixture()
    frame = _extract_frame(VIDEO_FIXTURE, tmp_path)

    llm = LLM(GEMMA_MODEL, think=False)

    from merceka_core.errors import GpuLockTimeout

    holder_done = asyncio.Event()
    competitor_saw_timeout = False

    async def holder():
      async with gpu_lock(timeout=60):
        # Kick off a real (short) inference while holding the lock, so
        # the competitor has a chance to race against the held lock.
        await llm.agenerate_with_resource("Describe in 3 words.", frame)
        holder_done.set()

    async def competitor():
      nonlocal competitor_saw_timeout
      # Wait for the holder to enter the lock region. We can't peek at
      # fcntl state so we give the event loop a tick to let `holder`
      # acquire first.
      await asyncio.sleep(0.05)
      try:
        async with gpu_lock(timeout=0.1):
          pass
      except GpuLockTimeout:
        competitor_saw_timeout = True

    await asyncio.gather(holder(), competitor())

    assert holder_done.is_set(), "holder never finished — lock may be stuck"
    assert competitor_saw_timeout, (
      "competitor acquired the lock while holder was inside gpu_lock — "
      "the lock is not providing mutual exclusion"
    )
    # Sanity: lock file exists on the XDG path (not tmpfs).
    assert GPU_LOCK_PATH.exists()
