"""Every paid LLM call writes one cost-ledger row (merceka_core.costs).

Transports are faked; the ledger is the per-test file conftest points
MERCEKA_COST_LEDGER at.
"""

import asyncio
import json

import pytest

import merceka_core.llm as llm_module
from merceka_core import costs
from merceka_core.llm import LLM

USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0012}


def _body(content: str | None = "hi", tool_calls=None, gen_id="gen-1"):
  message = {"role": "assistant", "content": content}
  if tool_calls:
    message["tool_calls"] = tool_calls
  return {"id": gen_id, "choices": [{"message": message}], "usage": dict(USAGE)}


def _tool_call_body(gen_id):
  return _body(
    content=None,
    gen_id=gen_id,
    tool_calls=[{
      "id": "call_0",
      "type": "function",
      "function": {"name": "lookup", "arguments": json.dumps({"key": "x"})},
    }],
  )


def _rows():
  path = costs.ledger_path()
  if not path.exists():
    return []
  return [json.loads(line) for line in path.read_text().splitlines()]


def lookup(key: str) -> str:
  """Look up a value.

  Args:
    key: What to look up.
  """
  return f"value_for_{key}"


class _FakeUrlopenResponse:
  def __init__(self, body):
    self._data = json.dumps(body).encode("utf-8")

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    return False

  def read(self, *_args):
    return self._data


def _install_urlopen(monkeypatch, bodies):
  queue = list(bodies)
  monkeypatch.setattr(
    llm_module, "urlopen", lambda request, timeout=None: _FakeUrlopenResponse(queue.pop(0)))


class _FakeAsyncResponse:
  def __init__(self, body):
    self._body = body

  def raise_for_status(self):
    return None

  def json(self):
    return self._body


def _install_async_client(monkeypatch, bodies):
  queue = list(bodies)

  class FakeAsyncClient:
    def __init__(self, *args, **kwargs):
      pass

    async def __aenter__(self):
      return self

    async def __aexit__(self, *exc):
      return False

    async def post(self, *_args, **_kwargs):
      return _FakeAsyncResponse(queue.pop(0))

  monkeypatch.setattr(llm_module.httpx, "AsyncClient", FakeAsyncClient)


@pytest.fixture(autouse=True)
def _openrouter_key(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


class TestOpenRouterLedger:
  def test_sync_row_carries_provider_cost_and_generation_id(self, monkeypatch):
    _install_urlopen(monkeypatch, [_body(gen_id="gen-sync")])
    assert LLM("openrouter/google/gemini-3-flash").generate("x") == "hi"
    [row] = _rows()
    assert row["source"] == "openrouter"
    assert row["model"] == "google/gemini-3-flash"
    assert row["usd"] == 0.0012 and row["usd_source"] == "provider"
    assert row["request_id"] == "gen-sync"

  def test_async_generate_writes_one_row(self, monkeypatch):
    _install_async_client(monkeypatch, [_body(gen_id="gen-async")])
    assert asyncio.run(LLM("openrouter/google/gemini-3-flash").agenerate("x")) == "hi"
    [row] = _rows()
    assert row["source"] == "openrouter"
    assert row["model"] == "google/gemini-3-flash"
    assert row["usd"] == 0.0012
    assert row["usage"]["prompt_tokens"] == 10
    assert row["request_id"] == "gen-async"

  def test_agenerate_with_resource_writes_one_row(self, monkeypatch, tmp_path):
    """mindweaver's screenshot path."""
    png = tmp_path / "shot.png"
    png.write_bytes(b"\x89PNG fake")
    _install_async_client(monkeypatch, [_body(gen_id="gen-resource")])
    llm = LLM("openrouter/anthropic/claude-sonnet-4-5")
    assert asyncio.run(llm.agenerate_with_resource("what is this?", png)) == "hi"
    [row] = _rows()
    assert row["model"] == "anthropic/claude-sonnet-4-5"
    assert row["request_id"] == "gen-resource"

  def test_async_row_is_written_before_parsing(self, monkeypatch):
    """A charged call whose body cannot be parsed is still metered."""
    body = {"id": "gen-broken", "usage": dict(USAGE)}  # no "choices"
    _install_async_client(monkeypatch, [body])
    with pytest.raises(KeyError):
      asyncio.run(LLM("openrouter/x").agenerate("x"))
    [row] = _rows()
    assert row["request_id"] == "gen-broken" and row["usd"] == 0.0012

  def test_sync_tool_loop_writes_a_row_per_round(self, monkeypatch):
    _install_urlopen(monkeypatch, [_tool_call_body("gen-r1"), _body("done", gen_id="gen-r2")])
    assert LLM("openrouter/x", tools=[lookup]).generate("x") == "done"
    assert [row["request_id"] for row in _rows()] == ["gen-r1", "gen-r2"]
    assert all(row["usd"] == 0.0012 for row in _rows())

  def test_async_tool_loop_writes_a_row_per_round(self, monkeypatch):
    _install_async_client(
      monkeypatch, [_tool_call_body("gen-a1"), _body("done", gen_id="gen-a2")])
    assert asyncio.run(LLM("openrouter/x", tools=[lookup]).agenerate("x")) == "done"
    assert [row["request_id"] for row in _rows()] == ["gen-a1", "gen-a2"]

  def test_tool_loop_row_is_written_before_parsing(self, monkeypatch):
    _install_urlopen(monkeypatch, [{"id": "gen-err", "usage": dict(USAGE)}])
    with pytest.raises(KeyError):
      LLM("openrouter/x", tools=[lookup]).generate("x")
    assert [row["request_id"] for row in _rows()] == ["gen-err"]
