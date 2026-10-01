"""Unit tests for the openrouter retry policy."""
from __future__ import annotations

import asyncio
import errno
import http.client
import json
import random
import socket
import urllib.error

import httpx
import pytest

import merceka_core.llm as llm_module
from merceka_core.llm import (
  LLM,
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


# --- Which OpenRouter failures are retried (sync urllib and async httpx) ---
#
# A retry is safe only when the request provably never reached the server, or
# the server answered with a retryable status. Anything that fails after the
# request was sent may have been processed and billed; retrying it would bill
# the call again.

_OK_BODY = {"choices": [{"message": {"content": "ok"}}], "usage": {}}


class _Response:
  def __enter__(self):
    return self

  def __exit__(self, *exc):
    return False

  def read(self, *_args):
    return json.dumps(_OK_BODY).encode()


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  monkeypatch.setattr(llm_module, "_retry_delay", lambda *_args, **_kwargs: 0.0)


def _sync_attempts(monkeypatch, error):
  """Attempts made by the sync path when every POST fails with ``error``."""
  attempts = []

  def fake_urlopen(*_args, **_kwargs):
    attempts.append(1)
    raise error

  monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)
  with pytest.raises(type(error)):
    LLM("openrouter/x")._openrouter_call([{"role": "user", "content": "q"}])
  return len(attempts)


def _http_error(code):
  return urllib.error.HTTPError("https://openrouter.ai", code, "err", {}, None)


SYNC_RETRIED = [
  pytest.param(
    urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED, "refused")), id="refused"),
  pytest.param(urllib.error.URLError(socket.gaierror(8, "nodename")), id="dns"),
  pytest.param(
    urllib.error.URLError(OSError(errno.ENETUNREACH, "unreachable")), id="network-unreachable"),
  pytest.param(_http_error(429), id="429"),
  pytest.param(_http_error(503), id="503"),
]

SYNC_NOT_RETRIED = [
  # getresponse() failures: the request was sent in full.
  pytest.param(TimeoutError("The read operation timed out"), id="read-timeout"),
  pytest.param(http.client.RemoteDisconnected("closed"), id="remote-disconnected"),
  pytest.param(ConnectionResetError(errno.ECONNRESET, "reset"), id="reset-after-send"),
  # urllib raises a connect timeout and a send timeout identically.
  pytest.param(urllib.error.URLError(TimeoutError("timed out")), id="connect-or-write-timeout"),
  pytest.param(_http_error(400), id="400"),
]


class TestSyncOpenRouterRetrySet:
  @pytest.mark.parametrize("error", SYNC_RETRIED)
  def test_retried(self, monkeypatch, error):
    assert _sync_attempts(monkeypatch, error) == _RETRY_MAX_ATTEMPTS

  @pytest.mark.parametrize("error", SYNC_NOT_RETRIED)
  def test_not_retried(self, monkeypatch, error):
    assert _sync_attempts(monkeypatch, error) == 1

  def test_recovers_after_a_refused_connection(self, monkeypatch):
    refused = ConnectionRefusedError(errno.ECONNREFUSED, "refused")
    outcomes = [urllib.error.URLError(refused), _Response()]

    def fake_urlopen(*_args, **_kwargs):
      outcome = outcomes.pop(0)
      if isinstance(outcome, Exception):
        raise outcome
      return outcome

    monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)
    assert LLM("openrouter/x")._openrouter_call([{"role": "user", "content": "q"}]) == "ok"


def _async_attempts(monkeypatch, error):
  """Attempts made by the async path when every POST fails with ``error``."""
  attempts = []

  class FakeAsyncClient:
    def __init__(self, *_args, **_kwargs):
      pass

    async def __aenter__(self):
      return self

    async def __aexit__(self, *exc):
      return False

    async def post(self, *_args, **_kwargs):
      attempts.append(1)
      raise error

  monkeypatch.setattr(llm_module.httpx, "AsyncClient", FakeAsyncClient)
  with pytest.raises(type(error)):
    asyncio.run(LLM("openrouter/x")._aopenrouter_call([{"role": "user", "content": "q"}]))
  return len(attempts)


def _status_error(code):
  request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
  response = httpx.Response(code, request=request)
  return httpx.HTTPStatusError(f"{code}", request=request, response=response)


ASYNC_RETRIED = [
  pytest.param(httpx.ConnectError("refused"), id="connect-error"),
  pytest.param(httpx.ConnectTimeout("connect timed out"), id="connect-timeout"),
  pytest.param(httpx.PoolTimeout("pool exhausted"), id="pool-timeout"),
  pytest.param(_status_error(429), id="429"),
  pytest.param(_status_error(503), id="503"),
]

ASYNC_NOT_RETRIED = [
  pytest.param(httpx.ReadTimeout("read timed out"), id="read-timeout"),
  pytest.param(httpx.WriteTimeout("write timed out"), id="write-timeout"),
  pytest.param(httpx.RemoteProtocolError("server disconnected"), id="remote-disconnected"),
  pytest.param(httpx.ReadError("reset"), id="read-error"),
  pytest.param(_status_error(400), id="400"),
]


class TestAsyncOpenRouterRetrySet:
  @pytest.mark.parametrize("error", ASYNC_RETRIED)
  def test_retried(self, monkeypatch, error):
    assert _async_attempts(monkeypatch, error) == _RETRY_MAX_ATTEMPTS

  @pytest.mark.parametrize("error", ASYNC_NOT_RETRIED)
  def test_not_retried(self, monkeypatch, error):
    assert _async_attempts(monkeypatch, error) == 1
