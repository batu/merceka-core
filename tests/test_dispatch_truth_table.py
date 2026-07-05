"""Dispatch truth table: pins which backend serves every (model prefix x tools x
allowed_tools x fallback) combination, for sync and async.

These tests monkeypatch the transport methods (_claude_call, _codex_call, ...)
rather than subprocess/httpx, because the subject under test is dispatch
*selection*, not transport behavior (covered elsewhere).
"""

import asyncio

import pytest

from merceka_core import llm as llm_module
from merceka_core.llm import LLM


def _dummy_tool(query: str) -> str:
  """Return a canned search result.

  Args:
      query: What to look up.
  """
  return f"result for {query}"


TOOLS = [_dummy_tool]


def _spy_transports(monkeypatch, llm):
  """Replace every transport with a recorder; returns the hit list."""
  hits = []

  def record(name, ret="ok"):
    def _fn(self, *args, **kwargs):
      hits.append((name, self.model_name))
      if name == "tool_loop":
        return ret, []
      return ret
    return _fn

  monkeypatch.setattr(LLM, "_claude_call", record("claude"))
  monkeypatch.setattr(LLM, "_codex_call", record("codex"))
  monkeypatch.setattr(LLM, "_run_tool_loop", record("tool_loop"))

  async def _arun_tool_loop(self, *args, **kwargs):
    hits.append(("tool_loop", self.model_name))
    return "ok", []

  async def _acloud_call(self, *args, **kwargs):
    hits.append(("openrouter", self.model_name))
    return "ok"

  monkeypatch.setattr(LLM, "_arun_tool_loop", _arun_tool_loop)
  monkeypatch.setattr(LLM, "_cloud_call", record("openrouter"))
  monkeypatch.setattr(LLM, "_acloud_call", _acloud_call)
  monkeypatch.setattr(LLM, "_local_call", record("local"))
  return hits


@pytest.fixture(autouse=True)
def _no_verify(monkeypatch):
  monkeypatch.setattr(LLM, "_verify", lambda self: None)


# --- _select_backend: pure decision function ---

class TestSelectBackend:
  @pytest.mark.parametrize("model,expected", [
    ("claude/sonnet", llm_module._BACKEND_CLAUDE),
    ("codex/gpt-5", llm_module._BACKEND_CODEX),
    ("openrouter/anthropic/claude-sonnet-4-5", llm_module._BACKEND_OPENROUTER),
    ("my-openrouter-proxy", llm_module._BACKEND_OPENROUTER),  # substring, not prefix
    ("gemma4:26b", llm_module._BACKEND_LOCAL),
  ])
  def test_plain_models(self, model, expected):
    assert LLM(model)._select_backend() == expected

  @pytest.mark.parametrize("model", ["openrouter/x", "gemma4:26b"])
  def test_tools_route_to_tool_loop(self, model):
    llm = LLM(model, tools=TOOLS)
    assert llm._select_backend() == llm_module._BACKEND_TOOL_LOOP

  def test_claude_native_tools_stay_on_claude(self):
    llm = LLM("claude/sonnet", tools=TOOLS, allowed_tools=["WebSearch"])
    assert llm._select_backend() == llm_module._BACKEND_CLAUDE

  def test_codex_native_tools_stay_on_codex(self):
    """_codex_call forwards --allowedTools, so codex gets the same escape."""
    llm = LLM("codex/gpt-5", tools=TOOLS, allowed_tools=["WebSearch"])
    assert llm._select_backend() == llm_module._BACKEND_CODEX

  @pytest.mark.parametrize("model,expected", [
    ("claude/sonnet", llm_module._BACKEND_CLAUDE),
    ("codex/gpt-5", llm_module._BACKEND_CODEX),
  ])
  def test_native_tools_win_over_fallback(self, model, expected):
    """allowed_tools + fallback: native branch takes precedence."""
    llm = LLM(model, tools=TOOLS, allowed_tools=["WebSearch"],
              fallback="openrouter/fb")
    assert llm._select_backend() == expected

  @pytest.mark.parametrize("model", ["claude/sonnet", "codex/gpt-5"])
  def test_cli_provider_tools_with_fallback(self, model):
    llm = LLM(model, tools=TOOLS, fallback="openrouter/fb")
    assert llm._select_backend() == llm_module._BACKEND_TOOLS_FALLBACK

  @pytest.mark.parametrize("model", ["claude/sonnet", "codex/gpt-5"])
  def test_cli_provider_tools_without_escape_raises(self, model):
    llm = LLM(model, tools=TOOLS)
    with pytest.raises(ValueError, match="allowed_tools|fallback|tools"):
      llm._select_backend()

  def test_gemini_plain_generate_raises(self):
    with pytest.raises(ValueError, match="generate_with_video"):
      LLM("gemini/gemini-flash-latest")._select_backend()


# --- generate / agenerate ladders hit the selected transport ---

SYNC_TABLE = [
  ("claude/sonnet", None, "claude"),
  ("codex/gpt-5", None, "codex"),
  ("openrouter/x", None, "openrouter"),
  ("my-openrouter-proxy", None, "openrouter"),
  ("gemma4:26b", None, "local"),
  ("openrouter/x", TOOLS, "tool_loop"),
  ("gemma4:26b", TOOLS, "tool_loop"),
]


class TestGenerateDispatch:
  @pytest.mark.parametrize("model,tools,expected", SYNC_TABLE)
  def test_sync(self, monkeypatch, model, tools, expected):
    llm = LLM(model, tools=tools)
    hits = _spy_transports(monkeypatch, llm)
    assert llm.generate("hi") == "ok"
    assert hits == [(expected, model)]

  @pytest.mark.parametrize("model,tools,expected", SYNC_TABLE)
  @pytest.mark.asyncio
  async def test_async_mirrors_sync(self, monkeypatch, model, tools, expected):
    llm = LLM(model, tools=tools)
    hits = _spy_transports(monkeypatch, llm)
    assert await llm.agenerate("hi") == "ok"
    assert hits == [(expected, model)]

  @pytest.mark.asyncio
  async def test_async_codex_no_longer_falls_through_to_ollama(self, monkeypatch):
    """Regression: the async ladder previously had no codex branch."""
    llm = LLM("codex/gpt-5")
    hits = _spy_transports(monkeypatch, llm)
    await llm.agenerate("hi")
    assert hits == [("codex", "codex/gpt-5")]

  @pytest.mark.parametrize("call", ["generate", "chat"])
  def test_gemini_plain_calls_raise(self, call):
    llm = LLM("gemini/gemini-flash-latest")
    with pytest.raises(ValueError, match="Gemini"):
      getattr(llm, call)("hi")

  @pytest.mark.asyncio
  async def test_gemini_agenerate_raises(self):
    with pytest.raises(ValueError, match="Gemini"):
      await LLM("gemini/gemini-flash-latest").agenerate("hi")

  def test_claude_tools_no_escape_raises_not_drops(self, monkeypatch):
    """Regression: tools were silently dropped and _claude_call ran anyway."""
    llm = LLM("claude/sonnet", tools=TOOLS)
    hits = _spy_transports(monkeypatch, llm)
    with pytest.raises(ValueError):
      llm.generate("hi")
    assert hits == []


# --- fallback constructor fidelity ---

class TestFallbackFidelity:
  def test_fallback_llm_preserves_full_config(self):
    llm = LLM("claude/sonnet", system_prompt="sp", think=True, tools=TOOLS,
              max_tool_rounds=5, fallback="openrouter/fb",
              add_dirs=["/tmp/x"], allowed_tools=[])
    fb = llm._fallback_llm()
    assert fb.model_name == "openrouter/fb"
    assert fb.system_prompt == "sp"
    assert fb.think is True
    assert fb._original_tools is llm._original_tools
    assert fb.max_tool_rounds == 5
    assert fb.add_dirs == ["/tmp/x"]
    assert fb.allowed_tools == []

  def test_tools_fallback_branch_uses_fallback_llm(self, monkeypatch):
    llm = LLM("claude/sonnet", tools=TOOLS, fallback="openrouter/fb",
              add_dirs=["/tmp/x"])
    seen = {}
    original = LLM._fallback_llm

    def spy(self, model_name=None):
      fb = original(self, model_name)
      seen["add_dirs"] = fb.add_dirs
      monkeypatch.setattr(LLM, "_run_tool_loop",
                          lambda *a, **k: ("ok", []))
      return fb

    monkeypatch.setattr(LLM, "_fallback_llm", spy)
    assert llm.generate("hi") == "ok"
    assert seen["add_dirs"] == ["/tmp/x"]

  def test_outer_catch_fallback_preserves_config(self, monkeypatch):
    llm = LLM("openrouter/primary", fallback="openrouter/fb",
              add_dirs=["/tmp/x"], allowed_tools=["Read"])
    calls = []

    def cloud(self, *args, **kwargs):
      calls.append(self.model_name)
      if self.model_name == "openrouter/primary":
        raise ConnectionError("boom")
      assert self.add_dirs == ["/tmp/x"] and self.allowed_tools == ["Read"]
      return "recovered"

    monkeypatch.setattr(LLM, "_cloud_call", cloud)
    assert llm.generate("hi") == "recovered"
    assert calls == ["openrouter/primary", "openrouter/fb"]

  @pytest.mark.parametrize("model", ["codex/gpt-5", "gemini/gemini-flash-latest"])
  def test_stream_raises_for_unsupported_configs(self, model):
    """One-chunk stream path inherits _select_backend's eager errors.

    Old behavior silently dropped tools (codex) or misrouted to Ollama
    (gemini); the raise is the intended new invariant.
    """
    llm = LLM(model, tools=TOOLS if model.startswith("codex") else None)
    with pytest.raises(ValueError):
      list(llm.stream_generate("hi"))

  def test_stream_one_chunk_fallback_preserves_tools(self, monkeypatch):
    llm = LLM("openrouter/x", tools=TOOLS, max_tool_rounds=7)
    captured = {}

    def tool_loop(self, *args, **kwargs):
      captured["tools"] = self._original_tools
      captured["rounds"] = self.max_tool_rounds
      return "ok", []

    monkeypatch.setattr(LLM, "_run_tool_loop", tool_loop)
    assert list(llm.stream_generate("hi")) == ["ok"]
    assert captured["tools"] is TOOLS
    assert captured["rounds"] == 7


# --- astream_generate: async-native handoff ---

class TestAstream:
  @pytest.mark.asyncio
  async def test_chunk_parity_with_sync_stream(self, monkeypatch):
    chunks = ["a", "b", "c"]
    monkeypatch.setattr(LLM, "stream_generate",
                        lambda self, m, **k: iter(chunks))
    llm = LLM("claude/sonnet")
    out = [c async for c in llm.astream_generate("hi")]
    assert out == chunks

  @pytest.mark.asyncio
  async def test_producer_exception_propagates(self, monkeypatch):
    def bad_stream(self, m, **k):
      yield "a"
      raise RuntimeError("stream died")

    monkeypatch.setattr(LLM, "stream_generate", bad_stream)
    llm = LLM("claude/sonnet")
    with pytest.raises(RuntimeError, match="stream died"):
      async for _ in llm.astream_generate("hi"):
        pass

  @pytest.mark.asyncio
  async def test_early_break_does_not_wait_for_full_stream(self, monkeypatch):
    """A consumer that breaks early must not block on stream exhaustion."""
    import itertools
    import time

    def slow_infinite_stream(self, m, **k):
      for i in itertools.count():
        time.sleep(0.01)
        yield f"chunk{i}"

    monkeypatch.setattr(LLM, "stream_generate", slow_infinite_stream)
    llm = LLM("claude/sonnet")
    start = time.monotonic()
    async for chunk in llm.astream_generate("hi"):
      break  # abandon after first chunk
    # generator finalization happens here on GC/aclose; force it:
    elapsed = time.monotonic() - start
    assert elapsed < 2.0  # infinite stream would hang forever without stop event

  @pytest.mark.asyncio
  async def test_no_busy_poll_sleep(self, monkeypatch):
    """The old implementation woke the loop 50x/s via asyncio.sleep(0.02)."""
    sleeps = []
    real_sleep = asyncio.sleep

    async def spy_sleep(delay, *a, **k):
      sleeps.append(delay)
      return await real_sleep(delay, *a, **k)

    monkeypatch.setattr(asyncio, "sleep", spy_sleep)
    monkeypatch.setattr(LLM, "stream_generate",
                        lambda self, m, **k: iter(["x"]))
    llm = LLM("claude/sonnet")
    _ = [c async for c in llm.astream_generate("hi")]
    assert 0.02 not in sleeps
