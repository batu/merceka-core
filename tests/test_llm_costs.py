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
    llm_module, "urlopen", lambda *_args, **_kwargs: _FakeUrlopenResponse(queue.pop(0)))


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
    def __init__(self, *_args, **_kwargs):
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


# --- Gemini (google-genai SDK) ---


def _usage_metadata(prompt=1000, image=400, text=100, thoughts=50):
  from google.genai import types

  return types.GenerateContentResponseUsageMetadata(
    prompt_token_count=prompt,
    candidates_token_count=image + text,
    thoughts_token_count=thoughts,
    total_token_count=prompt + image + text + thoughts,
    candidates_tokens_details=[
      types.ModalityTokenCount(modality=types.MediaModality.IMAGE, token_count=image),
      types.ModalityTokenCount(modality=types.MediaModality.TEXT, token_count=text),
    ],
  )


EXPECTED_USAGE = {
  "promptTokenCount": 1000,
  "candidatesTokenCount": 500,
  "thoughtsTokenCount": 50,
  "totalTokenCount": 1550,
  "candidatesTokensDetails": [
    {"modality": "IMAGE", "tokenCount": 400},
    {"modality": "TEXT", "tokenCount": 100},
  ],
}


def _gemini_response(text="ok", response_id="resp-1"):
  from types import SimpleNamespace

  return SimpleNamespace(
    text=text, usage_metadata=_usage_metadata(), response_id=response_id, candidates=[])


class _FakeGeminiClient:
  def __init__(self, response):
    from types import SimpleNamespace

    self.deleted = []
    self.models = SimpleNamespace(generate_content=lambda **_kwargs: response)
    self.files = SimpleNamespace(
      upload=lambda **_kwargs: SimpleNamespace(name="files/v", state=SimpleNamespace(name="ACTIVE")),
      delete=lambda name: self.deleted.append(name),
    )


@pytest.fixture
def gemini(monkeypatch):
  from merceka_core import llm_gemini

  monkeypatch.setattr(LLM, "_verify", lambda self: None)

  def install(response):
    monkeypatch.setattr(llm_gemini, "_gemini_client", lambda: _FakeGeminiClient(response))

  return install


class TestGeminiLedger:
  def test_image_call_writes_google_direct_row_with_camelcase_usage(self, gemini, tmp_path):
    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG fake")
    gemini(_gemini_response())
    assert LLM("gemini/gemini-flash-latest").generate_with_resource("x", png) == "ok"
    [row] = _rows()
    assert row["source"] == "google-direct"
    assert row["model"] == "google/gemini-flash-latest"
    assert row["usage"] == EXPECTED_USAGE
    assert row["request_id"] == "resp-1"
    assert row["usd"] is None  # no rate entry: tokens kept, price never invented

  def test_async_image_call_writes_row(self, gemini, tmp_path):
    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG fake")
    gemini(_gemini_response())
    llm = LLM("gemini/gemini-flash-latest")
    assert asyncio.run(llm.agenerate_with_resource("x", png)) == "ok"
    assert [row["model"] for row in _rows()] == ["google/gemini-flash-latest"]

  def test_video_call_writes_row(self, gemini, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    gemini(_gemini_response("described"))
    llm = LLM("gemini/gemini-flash-latest")
    assert llm.generate_with_video("x", video, poll_interval_s=0) == "described"
    [row] = _rows()
    assert row["source"] == "google-direct" and row["usage"] == EXPECTED_USAGE

  def test_search_grounding_writes_row(self, gemini):
    from merceka_core.llm_gemini import generate_with_search_grounding

    gemini(_gemini_response("grounded"))
    text, _ = asyncio.run(generate_with_search_grounding(prompt="x", model="gemini-2.5-pro"))
    assert text == "grounded"
    [row] = _rows()
    assert row["model"] == "google/gemini-2.5-pro" and row["usage"] == EXPECTED_USAGE

  def test_row_is_written_before_parsing(self, gemini, tmp_path):
    from pydantic import ValidationError

    from merceka_core.llm import OutputSchema

    class Label(OutputSchema):
      label: str

    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG fake")
    gemini(_gemini_response("not json"))
    with pytest.raises(ValidationError):
      LLM("gemini/gemini-flash-latest", output_schema=Label).generate_with_resource("x", png)
    assert len(_rows()) == 1

  def test_rate_table_prices_the_modality_split(self, gemini, tmp_path, monkeypatch):
    """The usage dict uses the names costs.py prices: image output and
    text+thinking output are split by candidatesTokensDetails."""
    rates = tmp_path / "rates.json"
    rates.write_text(json.dumps({"google/gemini-img": {
      "promptTokenCount": 1.0, "outputImageTokens": 10.0, "outputTextTokens": 2.0}}))
    monkeypatch.setenv("MERCEKA_RATES_PATH", str(rates))
    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG fake")
    gemini(_gemini_response())
    LLM("gemini/gemini-img").generate_with_resource("x", png)
    [row] = _rows()
    # 1000 prompt * $1/M + 400 image * $10/M + (100 text + 50 thoughts) * $2/M
    assert row["usd"] == pytest.approx(0.0053)
    assert row["usd_source"] == "rates"

  def test_attribute_bag_usage_is_converted(self, gemini, tmp_path):
    """A snake_case attribute object (not the SDK class) still lands camelCase."""
    from types import SimpleNamespace

    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG fake")
    usage = SimpleNamespace(prompt_token_count=258, candidates_token_count=3)
    gemini(SimpleNamespace(text="ok", usage_metadata=usage))
    LLM("gemini/gemini-flash-latest").generate_with_resource("x", png)
    [row] = _rows()
    assert row["usage"] == {"promptTokenCount": 258, "candidatesTokenCount": 3}
    assert "request_id" not in row
