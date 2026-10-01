"""Tool schemas generated from Python signatures (messages.tool_from_callable)."""

from typing import Optional

from merceka_core.messages import _python_type_to_json, tool_from_callable


def test_optional_hints_use_the_inner_type():
  """Regression: int | None, Optional[float] and list[str] | None became "string"."""
  assert _python_type_to_json(int | None) == "integer"
  assert _python_type_to_json(Optional[float]) == "number"
  assert _python_type_to_json(list[str] | None) == "array"
  assert _python_type_to_json(bool | None) == "boolean"


def test_real_unions_keep_the_string_fallback():
  assert _python_type_to_json(int | str) == "string"
  assert _python_type_to_json(int | str | None) == "string"


def test_optional_params_in_a_tool_schema():
  def search(query: str, limit: int | None = None, cutoff: "float | None" = None) -> str:
    """Search."""
    return f"{query}{limit}{cutoff}"

  params = tool_from_callable(search)["function"]["parameters"]
  assert params["properties"]["limit"]["type"] == "integer"
  assert params["properties"]["cutoff"]["type"] == "number"
  assert params["required"] == ["query"]
