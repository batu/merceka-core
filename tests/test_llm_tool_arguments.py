"""The Python tool loop survives malformed tool-call arguments.

A model can emit arguments that are not valid JSON (truncated or single-quoted).
json.loads used to raise out of the loop and lose the whole call. The loop now
reports the problem back to the model as the tool result, so it can retry, the
same way a failing tool handler is reported.
"""

import asyncio

import pytest

from merceka_core.llm import LLM

CALLS: list[dict] = []


def lookup(key: str) -> str:
  """Look up a value.

  Args:
    key: What to look up.
  """
  CALLS.append({"key": key})
  return f"value_for_{key}"


@pytest.fixture(autouse=True)
def _reset():
  CALLS.clear()


def _tool_call(arguments):
  return {
    "role": "assistant",
    "content": None,
    "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "lookup", "arguments": arguments}}],
  }


def _answers(first_arguments):
  """Model turns: a tool call with ``first_arguments``, then valid ones, then text."""
  return [
    _tool_call(first_arguments),
    _tool_call({"key": "x"}),
    {"role": "assistant", "content": "done", "tool_calls": None},
  ]


def test_sync_loop_reports_malformed_arguments_to_the_model(monkeypatch):
  turns = _answers('{"key": "x"')  # truncated JSON
  seen = []

  def raw(self, messages, **_kwargs):
    seen.append([dict(m) for m in messages])
    return turns.pop(0)

  monkeypatch.setattr(LLM, "_cloud_call_raw", raw)
  assert LLM("openrouter/x", tools=[lookup]).generate("q") == "done"
  tool_result = seen[1][-1]
  assert tool_result["role"] == "tool" and tool_result["tool_call_id"] == "c1"
  assert "not valid JSON" in tool_result["content"]
  assert CALLS == [{"key": "x"}]  # the handler ran only for the valid call


def test_async_loop_reports_malformed_arguments_to_the_model(monkeypatch):
  turns = _answers("{'key': 'x'}")  # single quotes
  seen = []

  async def raw(self, messages, **_kwargs):
    seen.append([dict(m) for m in messages])
    return turns.pop(0)

  monkeypatch.setattr(LLM, "_acloud_call_raw", raw)
  assert asyncio.run(LLM("openrouter/x", tools=[lookup]).agenerate("q")) == "done"
  assert "not valid JSON" in seen[1][-1]["content"]
  assert CALLS == [{"key": "x"}]


def test_raw_openrouter_call_keeps_malformed_arguments_for_the_loop(monkeypatch):
  import json

  import merceka_core.llm as llm_module

  body = {"choices": [{"message": _tool_call('{"key": ')}]}

  class Response:
    def __enter__(self):
      return self

    def __exit__(self, *exc):
      return False

    def read(self, *_args):
      return json.dumps(body).encode()

  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  monkeypatch.setattr(llm_module, "urlopen", lambda *_args, **_kwargs: Response())
  msg = LLM("openrouter/x", tools=[lookup])._cloud_call_raw([{"role": "user", "content": "q"}])
  assert msg["tool_calls"][0]["function"]["arguments"] == '{"key": '


def test_non_object_arguments_are_reported(monkeypatch):
  turns = _answers("[1, 2]")

  def raw(self, messages, **_kwargs):
    return turns.pop(0)

  monkeypatch.setattr(LLM, "_cloud_call_raw", raw)
  assert LLM("openrouter/x", tools=[lookup]).generate("q") == "done"
  assert CALLS == [{"key": "x"}]
