"""Unit tests for the openrouter retry policy."""
from __future__ import annotations

import random


from merceka_core.llm import (
  _RETRY_BASE_DELAY,
  _RETRY_MAX_ATTEMPTS,
  _RETRY_MAX_DELAY,
  _RETRY_STATUS_CODES,
  _retry_after_seconds,
  _retry_delay,
)


class TestRetryDelay:
  def test_exponential_growth(self):
    # With jitter=0, delays should grow 1, 2, 4, 8, 16, 32 (clamped).
    random.seed(0)
    d0 = _retry_delay(0)
    random.seed(0)
    d1 = _retry_delay(1)
    random.seed(0)
    d2 = _retry_delay(2)
    # Same seed → same jitter, so ordering is preserved.
    assert d0 < d1 < d2

  def test_clamped_at_max_delay(self):
    # attempt=10 would be 1024 * base + jitter; must clamp to 30 + jitter.
    for _ in range(50):
      d = _retry_delay(10)
      assert d <= _RETRY_MAX_DELAY + 1.0

  def test_respects_retry_after(self):
    # When retry_after is provided, it takes precedence (clamped at max).
    assert _retry_delay(0, retry_after=5.0) == 5.0
    assert _retry_delay(5, retry_after=100.0) == _RETRY_MAX_DELAY

  def test_jitter_varies(self):
    # Different seeds → different delays (non-zero jitter).
    delays = {_retry_delay(0) for _ in range(50)}
    assert len(delays) > 1


class TestRetryAfterParsing:
  def test_seconds_value(self):
    headers = {"Retry-After": "15"}
    assert _retry_after_seconds(headers) == 15.0

  def test_lowercase_variant(self):
    headers = {"retry-after": "3"}
    assert _retry_after_seconds(headers) == 3.0

  def test_absent(self):
    assert _retry_after_seconds({}) is None

  def test_non_numeric(self):
    # HTTP-date format not implemented — must return None, not crash.
    headers = {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}
    assert _retry_after_seconds(headers) is None


class TestRetryStatusCodes:
  def test_contains_expected(self):
    assert 408 in _RETRY_STATUS_CODES
    assert 425 in _RETRY_STATUS_CODES
    assert 429 in _RETRY_STATUS_CODES
    assert 500 in _RETRY_STATUS_CODES
    assert 502 in _RETRY_STATUS_CODES
    assert 503 in _RETRY_STATUS_CODES
    assert 504 in _RETRY_STATUS_CODES
    assert 529 in _RETRY_STATUS_CODES

  def test_excludes_non_retryable_4xx(self):
    for code in (400, 401, 403, 404, 422):
      assert code not in _RETRY_STATUS_CODES


class TestRetryConstants:
  def test_base_delay(self):
    assert _RETRY_BASE_DELAY == 1.0

  def test_max_delay(self):
    assert _RETRY_MAX_DELAY == 30.0

  def test_max_attempts(self):
    assert _RETRY_MAX_ATTEMPTS == 3
