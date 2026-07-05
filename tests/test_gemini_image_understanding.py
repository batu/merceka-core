"""Tests for Gemini Flash image understanding via generate_with_resource."""

import pytest
from pydantic import Field

from merceka_core import llm_gemini
from merceka_core.errors import VideoBackendError
from merceka_core.llm import LLM, OutputSchema
from merceka_core.retry import _RETRY_MAX_ATTEMPTS


class FakeResponse:
  def __init__(self, text):
    self.text = text


class FakeModels:
  def __init__(self, outcomes):
    self.outcomes = list(outcomes)
    self.calls = []

  def generate_content(self, **kwargs):
    self.calls.append(kwargs)
    outcome = self.outcomes.pop(0)
    if isinstance(outcome, Exception):
      raise outcome
    return outcome


class FakeClient:
  def __init__(self, models):
    self.models = models


@pytest.fixture(autouse=True)
def _no_verify(monkeypatch):
  monkeypatch.setattr(LLM, "_verify", lambda self: None)


@pytest.fixture
def png(tmp_path):
  p = tmp_path / "shot.png"
  p.write_bytes(b"\x89PNG fakebytes")
  return p


def install_client(monkeypatch, outcomes):
  models = FakeModels(outcomes)
  monkeypatch.setattr(llm_gemini, "_gemini_client", lambda: FakeClient(models))
  return models


def transient_503():
  exc = RuntimeError("503 unavailable")
  exc.status_code = 503
  return exc


class TestGeminiImageCall:
  def test_happy_path_sends_inline_bytes_and_parses(self, monkeypatch, png):
    models = install_client(monkeypatch, [FakeResponse("a red arrow")])
    llm = LLM("gemini/gemini-flash-latest", system_prompt="be terse")
    out = llm.generate_with_resource("what is this?", png)
    assert out == "a red arrow"
    call = models.calls[0]
    assert call["model"] == "gemini-flash-latest"  # prefix stripped
    part, message = call["contents"]
    assert message == "what is this?"
    assert part.inline_data.mime_type == "image/png"
    assert part.inline_data.data == b"\x89PNG fakebytes"
    assert call["config"].system_instruction == "be terse"

  def test_output_schema_parsed(self, monkeypatch, png):
    class Label(OutputSchema):
      label: str = Field(description="object label")

    models = install_client(monkeypatch, [FakeResponse('{"label": "arrow"}')])
    llm = LLM("gemini/gemini-flash-latest", output_schema=Label)
    out = llm.generate_with_resource("label it", png)
    assert isinstance(out, Label) and out.label == "arrow"
    assert models.calls  # went through the fake, not a real network call

  def test_missing_file_raises_before_client(self, monkeypatch, tmp_path):
    def boom():
      raise AssertionError("client should not be constructed")

    monkeypatch.setattr(llm_gemini, "_gemini_client", boom)
    llm = LLM("gemini/gemini-flash-latest")
    with pytest.raises(FileNotFoundError):
      llm.generate_with_resource("hi", tmp_path / "missing.png")

  def test_retries_on_503_then_succeeds(self, monkeypatch, png):
    sleeps = []
    monkeypatch.setattr(llm_gemini.time, "sleep", sleeps.append)
    models = install_client(
      monkeypatch, [transient_503(), FakeResponse("ok")])
    out = LLM("gemini/gemini-flash-latest").generate_with_resource("hi", png)
    assert out == "ok"
    assert len(models.calls) == 2
    assert len(sleeps) == 1

  def test_non_retryable_raises_video_backend_error(self, monkeypatch, png):
    exc = RuntimeError("400 bad request")
    exc.status_code = 400
    models = install_client(monkeypatch, [exc])
    with pytest.raises(VideoBackendError, match="Gemini image"):
      LLM("gemini/gemini-flash-latest").generate_with_resource("hi", png)
    assert len(models.calls) == 1  # no retry on 4xx

  def test_retries_exhausted(self, monkeypatch, png):
    monkeypatch.setattr(llm_gemini.time, "sleep", lambda _delay: None)
    models = install_client(
      monkeypatch, [transient_503() for _ in range(_RETRY_MAX_ATTEMPTS)])
    with pytest.raises(VideoBackendError):
      LLM("gemini/gemini-flash-latest").generate_with_resource("hi", png)
    assert len(models.calls) == _RETRY_MAX_ATTEMPTS

  def test_mime_fallback_for_unknown_suffix(self, monkeypatch, tmp_path):
    weird = tmp_path / "image.unknownext"
    weird.write_bytes(b"data")
    models = install_client(monkeypatch, [FakeResponse("ok")])
    LLM("gemini/gemini-flash-latest").generate_with_resource("hi", weird)
    part, _ = models.calls[0]["contents"]
    assert part.inline_data.mime_type == "application/octet-stream"

  @pytest.mark.asyncio
  async def test_async_mirrors_sync(self, monkeypatch, png):
    models = install_client(monkeypatch, [FakeResponse("async ok")])
    llm = LLM("gemini/gemini-flash-latest")
    out = await llm.agenerate_with_resource("hi", png)
    assert out == "async ok"
    assert models.calls[0]["model"] == "gemini-flash-latest"


class TestDispatch:
  def test_gemini_no_longer_falls_through_to_ollama(self, monkeypatch, png):
    """Regression: gemini/ generate_with_resource previously hit _local_call."""
    models = install_client(monkeypatch, [FakeResponse("ok")])

    def ollama_trap(*_args, **_kwargs):
      raise AssertionError("gemini model must not reach the Ollama path")

    monkeypatch.setattr(LLM, "_local_call", ollama_trap)
    out = LLM("gemini/gemini-flash-latest").generate_with_resource("hi", png)
    assert out == "ok" and models.calls
