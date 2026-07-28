"""Vulture whitelist — names that look unused but are consumed indirectly."""


class _Whitelist:
  # pytest fixtures are "used" via parameter injection, invisible to vulture.
  api_keys = None
  # __exit__(self, *exc) / __exit__(self, exc_type, exc, tb) signatures —
  # the parameters are required by the context-manager protocol.
  exc = None
  exc_type = None
  tb = None
  # Tool-function parameters in test_tool_calling.py: the params exist to
  # exercise schema generation from signatures, not to be read in the body.
  limit = None
  verbose = None
  items = None
  value = None
  # Tool-function parameter in test_claude_provider.py's delegation test —
  # exists to exercise schema generation from the signature.
  q = None


_Whitelist.api_keys
_Whitelist.q
_Whitelist.exc
_Whitelist.exc_type
_Whitelist.tb
_Whitelist.limit
_Whitelist.verbose
_Whitelist.items
_Whitelist.value

# --- Tool-function stub params in tests (names are part of the LLM tool schema) ---
q  # noqa
