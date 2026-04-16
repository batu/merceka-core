"""End-to-end Gemini video upload + generate.

Requires ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``) in env and network
access. Runs only with ``-m integration``.

To execute:
    GOOGLE_API_KEY=... uv run pytest tests/integration/test_llm_video.py -m integration -s
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from merceka_core.llm import LLM

pytestmark = pytest.mark.integration


# A <5MB test fixture is expected to live next to this file as
# ``fixture_small.mp4``. If it's missing, the test skips with a clear
# message — we don't want to ship a binary in a PR.
FIXTURE_PATH = Path(__file__).parent / "fixture_small.mp4"


def _require_env():
  if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    pytest.skip("GOOGLE_API_KEY / GEMINI_API_KEY not set")


def _require_fixture():
  if not FIXTURE_PATH.exists():
    pytest.skip(f"fixture video missing at {FIXTURE_PATH}")


class TestGeminiVideoRoundtrip:
  def test_generate_with_video_returns_string(self):
    _require_env()
    _require_fixture()

    llm = LLM("gemini/2.5-pro")
    result = llm.generate_with_video(
      "Describe what happens in this short video in one sentence.",
      FIXTURE_PATH,
      timeout_s=60.0,
      poll_interval_s=2.0,
    )
    assert isinstance(result, str)
    assert len(result) > 0

  @pytest.mark.asyncio
  async def test_agenerate_with_video_returns_string(self):
    _require_env()
    _require_fixture()

    llm = LLM("gemini/2.5-pro")
    result = await llm.agenerate_with_video(
      "Describe this video.",
      FIXTURE_PATH,
      timeout_s=60.0,
      poll_interval_s=2.0,
    )
    assert isinstance(result, str)
    assert len(result) > 0
