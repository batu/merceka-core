"""Tests for tool calling / agentic loop in merceka_core.llm."""

import asyncio
import json
import json as json_module

import pytest

import merceka_core.llm as llm_module
from merceka_core.llm import (
  LLM,
  OutputSchema,
  tool_from_callable,
  _python_type_to_json,
  _parse_param_docs,
)


# --- tool_from_callable tests ---


def test_tool_from_callable_basic():
  def greet(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"

  schema = tool_from_callable(greet)
  assert schema["type"] == "function"
  assert schema["function"]["name"] == "greet"
  assert schema["function"]["description"] == "Say hello to someone."
  params = schema["function"]["parameters"]
  assert params["properties"]["name"]["type"] == "string"
  assert params["required"] == ["name"]


def test_tool_from_callable_multiple_types():
  def search(query: str, limit: int = 5, verbose: bool = False) -> str:
    """Search for items.

    Args:
      query: The search query
      limit: Max results to return
      verbose: Whether to include details
    """
    return ""

  schema = tool_from_callable(search)
  props = schema["function"]["parameters"]["properties"]
  assert props["query"]["type"] == "string"
  assert props["query"]["description"] == "The search query"
  assert props["limit"]["type"] == "integer"
  assert props["limit"]["description"] == "Max results to return"
  assert props["verbose"]["type"] == "boolean"
  # Only query is required (limit and verbose have defaults)
  assert schema["function"]["parameters"]["required"] == ["query"]


def test_tool_from_callable_no_hints():
  def mystery(x, y="default"):
    """A mystery function."""
    pass

  schema = tool_from_callable(mystery)
  props = schema["function"]["parameters"]["properties"]
  assert props["x"]["type"] == "string"  # fallback
  assert props["y"]["type"] == "string"
  assert schema["function"]["parameters"]["required"] == ["x"]


def test_tool_from_callable_list_type():
  def process(items: list[str]) -> str:
    """Process items."""
    return ""

  schema = tool_from_callable(process)
  assert schema["function"]["parameters"]["properties"]["items"]["type"] == "array"


def test_tool_from_callable_no_docstring():
  def bare(x: int):
    pass

  schema = tool_from_callable(bare)
  assert schema["function"]["description"] == "bare"


def test_tool_from_callable_float_type():
  def calculate(value: float) -> str:
    """Calculate something."""
    return ""

  schema = tool_from_callable(calculate)
  assert schema["function"]["parameters"]["properties"]["value"]["type"] == "number"


# --- _python_type_to_json tests ---


def test_python_type_to_json():
  assert _python_type_to_json(str) == "string"
  assert _python_type_to_json(int) == "integer"
  assert _python_type_to_json(float) == "number"
  assert _python_type_to_json(bool) == "boolean"
  assert _python_type_to_json(list) == "array"
  assert _python_type_to_json(dict) == "string"  # unknown falls back


# --- _parse_param_docs tests ---


def test_parse_param_docs():
  doc = """Do something.

  Args:
    name: The name to use
    count: How many times
  """
  result = _parse_param_docs(doc)
  assert result["name"] == "The name to use"
  assert result["count"] == "How many times"


def test_parse_param_docs_empty():
  assert _parse_param_docs(None) == {}
  assert _parse_param_docs("No args section here.") == {}


# --- LLM tool init tests ---


def test_tools_schema_mutex():
  class MySchema(OutputSchema):
    x: str

  def dummy(q: str) -> str:
    return q

  with pytest.raises(ValueError, match="Cannot use both"):
    LLM("openrouter/test", tools=[dummy], output_schema=MySchema)


def test_tools_init_from_callable():
  def search(query: str) -> str:
    """Search."""
    return ""

  llm = LLM("openrouter/test", tools=[search])
  assert len(llm._tool_schemas) == 1
  assert llm._tool_schemas[0]["function"]["name"] == "search"
  assert "search" in llm._tool_handlers


def test_tools_init_from_raw_schema():
  raw_schema = {
    "type": "function",
    "function": {
      "name": "my_tool",
      "description": "A tool",
      "parameters": {"type": "object", "properties": {}},
    },
  }

  def handler(**kwargs):
    return "ok"

  llm = LLM("openrouter/test", tools=[(raw_schema, handler)])
  assert len(llm._tool_schemas) == 1
  assert llm._tool_handlers["my_tool"] is handler


# --- _execute_tool_call tests ---


def test_execute_tool_call():
  def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(int(a) + int(b))

  llm = LLM("openrouter/test", tools=[add])
  result = llm._execute_tool_call({
    "id": "call_0",
    "type": "function",
    "function": {"name": "add", "arguments": {"a": 3, "b": 4}},
  })
  assert result == "7"


def test_execute_tool_call_unknown():
  def dummy(x: str) -> str:
    return x

  llm = LLM("openrouter/test", tools=[dummy])
  result = llm._execute_tool_call({
    "id": "call_0",
    "type": "function",
    "function": {"name": "nonexistent", "arguments": {}},
  })
  assert "unknown tool" in result


def test_execute_tool_call_error():
  def fail(x: str) -> str:
    """Always fails."""
    raise ValueError("boom")

  llm = LLM("openrouter/test", tools=[fail])
  result = llm._execute_tool_call({
    "id": "call_0",
    "type": "function",
    "function": {"name": "fail", "arguments": {"x": "test"}},
  })
  assert "Error calling fail" in result
  assert "boom" in result


# --- Mock helpers ---


class _FakeHttpResponse:
  def __init__(self, body: dict):
    self._body = body

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc, tb):
    return False

  def read(self):
    return json.dumps(self._body).encode("utf-8")


class _FakeAsyncHttpResponse:
  def __init__(self, body: dict):
    self._body = body

  def raise_for_status(self):
    return None

  def json(self):
    return self._body


def _make_api_response(content=None, tool_calls=None):
  msg = {"role": "assistant", "content": content}
  if tool_calls:
    msg["tool_calls"] = tool_calls
  return {"choices": [{"message": msg}]}


# --- Tool loop tests ---


def test_tool_loop_single_round(monkeypatch):
  """LLM calls tool once, then returns final text."""
  call_count = 0

  def fake_urlopen(request, timeout=120):
    nonlocal call_count
    call_count += 1
    if call_count == 1:
      return _FakeHttpResponse(_make_api_response(
        tool_calls=[{
          "id": "call_0",
          "type": "function",
          "function": {"name": "lookup", "arguments": json.dumps({"key": "foo"})},
        }],
      ))
    else:
      return _FakeHttpResponse(_make_api_response(content="The answer is 42"))

  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)

  def lookup(key: str) -> str:
    """Look up a value."""
    return f"value_for_{key}"

  llm = LLM("openrouter/test", tools=[lookup])
  result = llm.generate("What is foo?")
  assert result == "The answer is 42"
  assert call_count == 2


def test_tool_loop_multi_round(monkeypatch):
  """LLM calls tools twice before final answer."""
  call_count = 0

  def fake_urlopen(request, timeout=120):
    nonlocal call_count
    call_count += 1
    if call_count <= 2:
      return _FakeHttpResponse(_make_api_response(
        tool_calls=[{
          "id": f"call_{call_count}",
          "type": "function",
          "function": {"name": "search", "arguments": json.dumps({"q": f"round_{call_count}"})},
        }],
      ))
    else:
      return _FakeHttpResponse(_make_api_response(content="Final answer"))

  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)

  def search(q: str) -> str:
    """Search."""
    return f"result_{q}"

  llm = LLM("openrouter/test", tools=[search])
  result = llm.generate("Tell me about X")
  assert result == "Final answer"
  assert call_count == 3


def test_tool_loop_max_iterations(monkeypatch):
  """Tool loop raises RuntimeError when max rounds exceeded."""
  def fake_urlopen(request, timeout=120):
    return _FakeHttpResponse(_make_api_response(
      tool_calls=[{
        "id": "call_0",
        "type": "function",
        "function": {"name": "infinite", "arguments": json.dumps({"x": "loop"})},
      }],
    ))

  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)

  def infinite(x: str) -> str:
    """Never stops."""
    return "more"

  llm = LLM("openrouter/test", tools=[infinite], max_tool_rounds=3)
  with pytest.raises(RuntimeError, match="exceeded 3 rounds"):
    llm.generate("go")


def test_chat_preserves_tool_history(monkeypatch):
  """chat() stores tool call trace in self.messages."""
  call_count = 0

  def fake_urlopen(request, timeout=120):
    nonlocal call_count
    call_count += 1
    if call_count == 1:
      return _FakeHttpResponse(_make_api_response(
        tool_calls=[{
          "id": "call_0",
          "type": "function",
          "function": {"name": "lookup", "arguments": json.dumps({"key": "x"})},
        }],
      ))
    else:
      return _FakeHttpResponse(_make_api_response(content="Done"))

  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)

  def lookup(key: str) -> str:
    """Look up."""
    return "found"

  llm = LLM("openrouter/test", tools=[lookup])
  result = llm.chat("find x")
  assert result == "Done"

  # Messages should contain: system, user, assistant(tool_call), tool(result), assistant(final)
  roles = [m["role"] for m in llm.messages]
  assert roles == ["system", "user", "assistant", "tool", "assistant"]
  assert llm.messages[2].get("tool_calls") is not None
  assert llm.messages[3]["content"] == "found"
  assert llm.messages[4]["content"] == "Done"


def test_tool_loop_no_tools_direct_response(monkeypatch):
  """When LLM responds without tool calls, return immediately."""
  def fake_urlopen(request, timeout=120):
    return _FakeHttpResponse(_make_api_response(content="Direct answer"))

  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)

  def unused_tool(x: str) -> str:
    """Never called."""
    return ""

  llm = LLM("openrouter/test", tools=[unused_tool])
  result = llm.generate("simple question")
  assert result == "Direct answer"


def test_tool_sends_schemas_in_payload(monkeypatch):
  """Verify tool schemas are included in the API request."""
  captured: dict = {}

  def fake_urlopen(request, timeout=120):
    captured["payload"] = json.loads(request.data.decode("utf-8"))
    return _FakeHttpResponse(_make_api_response(content="ok"))

  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)

  def my_tool(query: str) -> str:
    """Search for something.

    Args:
      query: The search query
    """
    return ""

  llm = LLM("openrouter/test", tools=[my_tool])
  llm.generate("test")

  assert "tools" in captured["payload"]
  tools = captured["payload"]["tools"]
  assert len(tools) == 1
  assert tools[0]["function"]["name"] == "my_tool"
  assert tools[0]["function"]["parameters"]["properties"]["query"]["type"] == "string"


# --- Async tool loop tests ---


@pytest.mark.asyncio
async def test_agenerate_with_tools(monkeypatch):
  """agenerate routes through async tool loop when tools are set."""
  call_count = 0

  class FakeAsyncClient:
    def __init__(self, timeout=120.0):
      pass

    async def __aenter__(self):
      return self

    async def __aexit__(self, exc_type, exc, tb):
      return False

    async def post(self, url, headers=None, json=None):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        body = _make_api_response(
          tool_calls=[{
            "id": "call_0",
            "type": "function",
            "function": {"name": "fetch", "arguments": json_module.dumps({"url": "test"})},
          }],
        )
      else:
        body = _make_api_response(content="Fetched result")
      return _FakeAsyncHttpResponse(body)

  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  monkeypatch.setattr(llm_module.httpx, "AsyncClient", FakeAsyncClient)

  def fetch(url: str) -> str:
    """Fetch a URL."""
    return f"content_from_{url}"

  llm = LLM("openrouter/test", tools=[fetch])
  result = await llm.agenerate("get test")
  assert result == "Fetched result"
  assert call_count == 2


@pytest.mark.asyncio
async def test_agenerate_with_async_tool_handler(monkeypatch):
  """Async tool handlers are awaited directly, not wrapped in to_thread."""
  call_count = 0
  handler_was_awaited = False

  class FakeAsyncClient:
    def __init__(self, timeout=120.0):
      pass

    async def __aenter__(self):
      return self

    async def __aexit__(self, exc_type, exc, tb):
      return False

    async def post(self, url, headers=None, json=None):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        body = _make_api_response(
          tool_calls=[{
            "id": "call_0",
            "type": "function",
            "function": {"name": "async_search", "arguments": json_module.dumps({"q": "test"})},
          }],
        )
      else:
        body = _make_api_response(content="Async result")
      return _FakeAsyncHttpResponse(body)

  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  monkeypatch.setattr(llm_module.httpx, "AsyncClient", FakeAsyncClient)

  async def async_search(q: str) -> str:
    """Async search."""
    nonlocal handler_was_awaited
    handler_was_awaited = True
    return f"found_{q}"

  llm = LLM("openrouter/test", tools=[async_search])
  result = await llm.agenerate("search for test")
  assert result == "Async result"
  assert handler_was_awaited


@pytest.mark.asyncio
async def test_agenerate_max_rounds_async(monkeypatch):
  """Async tool loop raises RuntimeError on max rounds exceeded."""

  class FakeAsyncClient:
    def __init__(self, timeout=120.0):
      pass

    async def __aenter__(self):
      return self

    async def __aexit__(self, exc_type, exc, tb):
      return False

    async def post(self, url, headers=None, json=None):
      return _FakeAsyncHttpResponse(_make_api_response(
        tool_calls=[{
          "id": "call_0",
          "type": "function",
          "function": {"name": "loop_tool", "arguments": json_module.dumps({"x": "y"})},
        }],
      ))

  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  monkeypatch.setattr(llm_module.httpx, "AsyncClient", FakeAsyncClient)

  def loop_tool(x: str) -> str:
    """Loops forever."""
    return "again"

  llm = LLM("openrouter/test", tools=[loop_tool], max_tool_rounds=2)
  with pytest.raises(RuntimeError, match="exceeded 2 rounds"):
    await llm.agenerate("go")


# --- Edge case tests ---


def test_tool_from_callable_multiline_docstring():
  """First paragraph of docstring is used as description, not the whole thing."""
  def complex_fn(a: str, b: int = 0) -> str:
    """Do the first thing.

    This is a longer explanation that should not
    be included in the description.

    Args:
      a: The first argument
      b: The second argument
    """
    return ""

  schema = tool_from_callable(complex_fn)
  assert schema["function"]["description"] == "Do the first thing."
  assert schema["function"]["parameters"]["properties"]["a"]["description"] == "The first argument"
  assert schema["function"]["parameters"]["properties"]["b"]["description"] == "The second argument"


def test_tool_from_callable_no_params():
  """Function with no parameters produces empty properties."""
  def noop() -> str:
    """Do nothing."""
    return ""

  schema = tool_from_callable(noop)
  assert schema["function"]["parameters"]["properties"] == {}
  assert "required" not in schema["function"]["parameters"]


def test_execute_tool_call_string_arguments():
  """Arguments passed as JSON string (OpenRouter format) are deserialized."""
  def echo(msg: str) -> str:
    """Echo."""
    return msg

  llm = LLM("openrouter/test", tools=[echo])
  result = llm._execute_tool_call({
    "id": "call_0",
    "type": "function",
    "function": {"name": "echo", "arguments": '{"msg": "hello"}'},
  })
  assert result == "hello"


def test_multiple_tools_registered():
  """Multiple tools are all registered and dispatchable."""
  def tool_a(x: str) -> str:
    """Tool A."""
    return f"a:{x}"

  def tool_b(y: int) -> str:
    """Tool B."""
    return f"b:{y}"

  llm = LLM("openrouter/test", tools=[tool_a, tool_b])
  assert len(llm._tool_schemas) == 2
  assert {s["function"]["name"] for s in llm._tool_schemas} == {"tool_a", "tool_b"}

  assert llm._execute_tool_call({
    "id": "c1", "type": "function",
    "function": {"name": "tool_a", "arguments": {"x": "test"}},
  }) == "a:test"

  assert llm._execute_tool_call({
    "id": "c2", "type": "function",
    "function": {"name": "tool_b", "arguments": {"y": 42}},
  }) == "b:42"


def test_tool_loop_multiple_tool_calls_in_single_response(monkeypatch):
  """LLM returns multiple tool_calls in one response — all are executed."""
  call_count = 0

  def fake_urlopen(request, timeout=120):
    nonlocal call_count
    call_count += 1
    if call_count == 1:
      return _FakeHttpResponse(_make_api_response(
        tool_calls=[
          {"id": "call_0", "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"key": "a"})}},
          {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"key": "b"})}},
        ],
      ))
    else:
      return _FakeHttpResponse(_make_api_response(content="Combined answer"))

  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)

  results_seen = []

  def lookup(key: str) -> str:
    """Look up."""
    results_seen.append(key)
    return f"val_{key}"

  llm = LLM("openrouter/test", tools=[lookup])
  result = llm.generate("find a and b")
  assert result == "Combined answer"
  assert results_seen == ["a", "b"]
  assert call_count == 2


def test_chat_multi_turn_preserves_tool_context(monkeypatch):
  """Second chat() call sees the full history from the first including tool traces."""
  call_count = 0

  def fake_urlopen(request, timeout=120):
    nonlocal call_count
    call_count += 1
    payload = json.loads(request.data.decode("utf-8"))
    if call_count == 1:
      # First call: tool call
      return _FakeHttpResponse(_make_api_response(
        tool_calls=[{
          "id": "call_0", "type": "function",
          "function": {"name": "get_info", "arguments": json.dumps({"topic": "x"})},
        }],
      ))
    elif call_count == 2:
      # Second call: final answer for first chat()
      return _FakeHttpResponse(_make_api_response(content="First answer"))
    else:
      # Third call: second chat() — verify it has the full history
      messages = payload["messages"]
      roles = [m["role"] for m in messages]
      # system, user1, assistant(tool_call), tool, assistant(answer), user2
      assert "tool" in roles, f"Expected 'tool' in message roles, got {roles}"
      return _FakeHttpResponse(_make_api_response(content="Second answer"))

  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)

  def get_info(topic: str) -> str:
    """Get info."""
    return f"info_about_{topic}"

  llm = LLM("openrouter/test", tools=[get_info])
  first = llm.chat("tell me about x")
  assert first == "First answer"

  second = llm.chat("follow up question")
  assert second == "Second answer"
  assert call_count == 3


def test_tool_loop_empty_content_returns_empty_string(monkeypatch):
  """When final response has no content (None), return empty string."""
  def fake_urlopen(request, timeout=120):
    return _FakeHttpResponse(_make_api_response(content=None))

  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)

  def tool(x: str) -> str:
    """A tool."""
    return x

  llm = LLM("openrouter/test", tools=[tool])
  result = llm.generate("test")
  assert result == ""


def test_cloud_call_raw_normalizes_string_arguments(monkeypatch):
  """_cloud_call_raw deserializes JSON string arguments into dicts."""
  def fake_urlopen(request, timeout=120):
    return _FakeHttpResponse({
      "choices": [{
        "message": {
          "role": "assistant",
          "content": None,
          "tool_calls": [{
            "id": "call_0",
            "type": "function",
            "function": {
              "name": "test_fn",
              "arguments": '{"key": "value"}',
            },
          }],
        },
      }],
    })

  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)

  def test_fn(key: str) -> str:
    """Test."""
    return key

  llm = LLM("openrouter/test", tools=[test_fn])
  msg = llm._cloud_call_raw([{"role": "user", "content": "test"}])
  assert msg["tool_calls"][0]["function"]["arguments"] == {"key": "value"}


def test_no_tools_generate_still_works(monkeypatch):
  """LLM without tools uses the normal non-tool code path."""
  def fake_urlopen(request, timeout=120):
    payload = json.loads(request.data.decode("utf-8"))
    assert "tools" not in payload
    return _FakeHttpResponse(_make_api_response(content="plain response"))

  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)

  llm = LLM("openrouter/test")
  result = llm.generate("hello")
  assert result == "plain response"


def test_parse_param_docs_multiline_description():
  """Param description that spans multiple lines is joined."""
  doc = """Do stuff.

  Args:
    name: This is a very long description
      that continues on the next line
    count: Simple description
  """
  result = _parse_param_docs(doc)
  assert "very long description" in result["name"]
  assert "continues on the next line" in result["name"]
  assert result["count"] == "Simple description"
