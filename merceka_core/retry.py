"""Retry policy for transient HTTP failures on cloud provider calls."""

import random

_RETRY_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504, 529})
_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 30.0
_RETRY_MAX_ATTEMPTS = 3


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
