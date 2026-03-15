"""Tests for structured LLM responses."""

import asyncio
import json

from pydantic import BaseModel, ConfigDict

import merceka_core.llm as llm_module
from merceka_core.llm import LLM, OutputSchema


class PlainStructuredResponse(BaseModel):
  model_config = ConfigDict(extra="forbid")

  summary: str
  commands: list[str]


class ContentOptionalResponse(OutputSchema):
  model_config = ConfigDict(extra="forbid")

  summary: str


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


def test_parse_response_accepts_dict_for_structured_schema():
  llm = LLM("openrouter/test-model", output_schema=PlainStructuredResponse)

  parsed = llm._parse_response({"summary": "ok", "commands": ["a"]})

  assert parsed.summary == "ok"
  assert parsed.commands == ["a"]


def test_output_schema_content_is_optional():
  payload = ContentOptionalResponse(summary="ok")

  assert payload.content is None
  assert payload.summary == "ok"


def test_chat_uses_json_when_structured_response_has_no_content(monkeypatch):
  llm = LLM("openrouter/test-model", output_schema=PlainStructuredResponse)
  monkeypatch.setattr(llm, "_cloud_call", lambda messages, **kwargs: PlainStructuredResponse(summary="ok", commands=[]))

  llm.chat("hello")

  assert llm.messages[-1]["role"] == "assistant"
  assert '"summary":"ok"' in llm.messages[-1]["content"]


def test_openrouter_structured_requests_send_json_schema(monkeypatch):
  captured: dict = {}

  def fake_urlopen(request, timeout=120):
    captured["headers"] = dict(request.header_items())
    captured["payload"] = json.loads(request.data.decode("utf-8"))
    return _FakeHttpResponse(
      {
        "choices": [
          {
            "message": {
              "content": {
                "summary": "ok",
                "commands": [],
              }
            }
          }
        ]
      }
    )

  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "http://localhost:5188")
  monkeypatch.setenv("OPENROUTER_X_TITLE", "Contexto Test")
  monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)

  llm = LLM("openrouter/google/gemini-3-flash-preview", output_schema=PlainStructuredResponse)
  parsed = llm.generate("hello", max_tokens=120)

  assert parsed.summary == "ok"
  assert captured["payload"]["model"] == "google/gemini-3-flash-preview"
  assert captured["payload"]["provider"]["require_parameters"] is True
  assert captured["payload"]["response_format"]["type"] == "json_schema"
  assert captured["payload"]["response_format"]["json_schema"]["strict"] is True
  assert captured["payload"]["plugins"] == [{"id": "response-healing"}]


def test_openrouter_async_structured_requests_send_json_schema(monkeypatch):
  captured: dict = {}

  class FakeAsyncClient:
    def __init__(self, timeout=120.0):
      captured["timeout"] = timeout

    async def __aenter__(self):
      return self

    async def __aexit__(self, exc_type, exc, tb):
      return False

    async def post(self, url, headers=None, json=None):
      captured["url"] = url
      captured["headers"] = dict(headers or {})
      captured["payload"] = json
      return _FakeAsyncHttpResponse(
        {
          "choices": [
            {
              "message": {
                "content": {
                  "summary": "ok",
                  "commands": [],
                }
              }
            }
          ]
        }
      )

  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "http://localhost:5188")
  monkeypatch.setenv("OPENROUTER_X_TITLE", "Contexto Test")
  monkeypatch.setattr(llm_module.httpx, "AsyncClient", FakeAsyncClient)

  llm = LLM("openrouter/google/gemini-3-flash-preview", output_schema=PlainStructuredResponse)
  parsed = asyncio.run(llm.agenerate("hello", max_tokens=120))

  assert parsed.summary == "ok"
  assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
  assert captured["payload"]["model"] == "google/gemini-3-flash-preview"
  assert captured["payload"]["provider"]["require_parameters"] is True
  assert captured["payload"]["response_format"]["type"] == "json_schema"
  assert captured["payload"]["response_format"]["json_schema"]["strict"] is True
  assert captured["payload"]["plugins"] == [{"id": "response-healing"}]
