"""LLM.agenerate_batch: a failure stops new paid calls and loses nothing it paid for.

Regression (review R9): one failure raised at once, threw away every result,
and the remaining calls, queued ones included, kept running and billing in the
background with nobody awaiting them.
"""

import asyncio

import pytest

from merceka_core.llm import LLM


@pytest.fixture
def calls(monkeypatch):
  """agenerate fake: item "bad" fails at once, the rest take 10 ms per item."""
  started = []
  finished = []

  async def fake_agenerate(self, message, **_kwargs):
    started.append(message)
    if message == "bad":
      raise ValueError("parse failure on bad")
    await asyncio.sleep(0.01)
    finished.append(message)
    return f"ok-{message}"

  monkeypatch.setattr(LLM, "agenerate", fake_agenerate)
  return started, finished


def test_failure_starts_no_new_calls_and_waits_for_in_flight_ones(calls):
  started, finished = calls
  messages = ["a", "bad", "b", "c", "d", "e"]
  with pytest.raises(ValueError, match="parse failure"):
    asyncio.run(LLM("openrouter/x").agenerate_batch(messages, concurrency=2, show_progress=False))
  # "a" was in flight when "bad" failed: it finished (and was metered) before
  # the raise. The queued items never started.
  assert started == ["a", "bad"]
  assert finished == ["a"]


def test_return_exceptions_keeps_every_paid_result(calls):
  started, _ = calls
  messages = ["a", "bad", "b"]
  results = asyncio.run(LLM("openrouter/x").agenerate_batch(
    messages, concurrency=2, show_progress=False, return_exceptions=True))
  assert results[0] == "ok-a" and results[2] == "ok-b"
  assert isinstance(results[1], ValueError)
  assert started == messages


def test_success_returns_results_in_input_order(calls):
  results = asyncio.run(LLM("openrouter/x").agenerate_batch(
    ["a", "b", "c"], concurrency=3, show_progress=False))
  assert results == ["ok-a", "ok-b", "ok-c"]


def test_progress_bar_path_has_the_same_failure_semantics(calls):
  started, _ = calls
  with pytest.raises(ValueError):
    asyncio.run(LLM("openrouter/x").agenerate_batch(
      ["a", "bad", "b", "c"], concurrency=2, show_progress=True))
  assert started == ["a", "bad"]
