"""LLM(timeout=) and the per-call timeout= kwarg set the OpenRouter HTTP timeout.

Before: urlopen/httpx always used 120 s, and a per-call timeout= was sent to
OpenRouter inside the JSON request body.
"""

import asyncio
import json

import pytest

import merceka_core.llm as llm_module
from merceka_core.llm import LLM

_BODY = {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}}


def lookup(key: str) -> str:
  """Look up a value.

  Args:
    key: What to look up.
  """
  return key


class _Response:
  def __enter__(self):
    return self

  def __exit__(self, *exc):
    return False

  def read(self, *_args):
    return json.dumps(_BODY).encode()


@pytest.fixture
def sync_http(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  seen = []

  def fake_urlopen(request, timeout=None):
    seen.append({"timeout": timeout, "payload": json.loads(request.data)})
    return _Response()

  monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)
  return seen


@pytest.fixture
def async_http(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  seen = []

  class FakeResponse:
    def raise_for_status(self):
      return None

    def json(self):
      return _BODY

  class FakeAsyncClient:
    def __init__(self, timeout=None):
      self._timeout = timeout

    async def __aenter__(self):
      return self

    async def __aexit__(self, *exc):
      return False

    async def post(self, _url, headers=None, json=None):
      del headers
      seen.append({"timeout": self._timeout, "payload": json})
      return FakeResponse()

  monkeypatch.setattr(llm_module.httpx, "AsyncClient", FakeAsyncClient)
  return seen


CASES = [
  pytest.param(None, {}, 120, id="default"),
  pytest.param(900, {}, 900, id="instance"),
  pytest.param(900, {"timeout": 30}, 30, id="per-call-wins"),
]


class TestOpenRouterHttpTimeout:
  @pytest.mark.parametrize("instance,call,expected", CASES)
  def test_sync(self, sync_http, instance, call, expected):
    LLM("openrouter/x", timeout=instance).generate("q", **call)
    assert sync_http[0]["timeout"] == expected
    assert "timeout" not in sync_http[0]["payload"]

  @pytest.mark.parametrize("instance,call,expected", CASES)
  def test_async(self, async_http, instance, call, expected):
    asyncio.run(LLM("openrouter/x", timeout=instance).agenerate("q", **call))
    assert async_http[0]["timeout"] == expected
    assert "timeout" not in async_http[0]["payload"]

  @pytest.mark.parametrize("instance,call,expected", CASES)
  def test_sync_tool_loop(self, sync_http, instance, call, expected):
    LLM("openrouter/x", tools=[lookup], timeout=instance).generate("q", **call)
    assert sync_http[0]["timeout"] == expected
    assert "timeout" not in sync_http[0]["payload"]

  @pytest.mark.parametrize("instance,call,expected", CASES)
  def test_async_tool_loop(self, async_http, instance, call, expected):
    asyncio.run(LLM("openrouter/x", tools=[lookup], timeout=instance).agenerate("q", **call))
    assert async_http[0]["timeout"] == expected
    assert "timeout" not in async_http[0]["payload"]

  def test_resource_call(self, sync_http, tmp_path):
    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG")
    LLM("openrouter/x", timeout=45).generate_with_resource("q", png)
    assert sync_http[0]["timeout"] == 45
    assert "timeout" not in sync_http[0]["payload"]
