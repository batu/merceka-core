"""Retry policy for transient HTTP failures on cloud provider calls."""

import errno
import random
import socket
import urllib.error

import httpx

_RETRY_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504, 529})
_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 30.0
_RETRY_MAX_ATTEMPTS = 3

# Transport failures are retried only when the request provably never reached
# the server: no connection was established. A failure after the request went
# out (read timeout, reset, disconnect) may follow a call the provider already
# processed and billed, and a retry would bill it again with only the last
# attempt metered, so those propagate to the caller (and its fallback). A write
# timeout means the request was partly sent, so it is not retried either.
_RETRY_HTTPX_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)

# socket errnos meaning the TCP connection was never established.
_NOT_CONNECTED_ERRNOS = frozenset({
  errno.ECONNREFUSED,
  errno.ENETUNREACH,
  errno.EHOSTUNREACH,
  errno.ENETDOWN,
  errno.EHOSTDOWN,
  errno.EADDRNOTAVAIL,
})


def _urlerror_never_sent(exc: urllib.error.URLError) -> bool:
  """True when urllib failed before any connection existed (DNS, refused, unreachable).

  urllib wraps connect and send errors alike in URLError and raises a connect
  timeout exactly like a send timeout, so a timeout is never treated as unsent.
  """
  reason = exc.reason
  if isinstance(reason, (socket.gaierror, ConnectionRefusedError)):
    return True
  return isinstance(reason, OSError) and reason.errno in _NOT_CONNECTED_ERRNOS


def _retry_delay(attempt: int, retry_after: float | None = None) -> float:
  """Exponential backoff with jitter, honoring Retry-After."""
  if retry_after is not None:
    return min(retry_after, _RETRY_MAX_DELAY)
  base = min(_RETRY_BASE_DELAY * (2 ** attempt), _RETRY_MAX_DELAY)
  return base + random.uniform(0, 1.0)


def _retry_after_seconds(headers) -> float | None:
  """Parse a Retry-After header value to seconds, or None."""
  value = None
  if hasattr(headers, "get"):
    value = headers.get("Retry-After") or headers.get("retry-after")
  if not value:
    return None
  try:
    return float(value)
  except (TypeError, ValueError):
    return None
