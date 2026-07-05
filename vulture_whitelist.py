"""Vulture whitelist — names that look unused but are consumed indirectly."""


class _Whitelist:
  # pytest fixtures are "used" via parameter injection, invisible to vulture.
  api_keys = None
  # __exit__(self, *exc) signature — exc is required by the protocol.
  exc = None


_Whitelist.api_keys
_Whitelist.exc
