"""The fallback cascade: at most one fallback run, with kwargs its transport accepts.

Per-call kwargs are written for the primary. A fallback on another transport
must not receive transport controls it does not understand: Ollama rejects
unknown kwargs with a TypeError, and OpenRouter would put them in the request
body. Transports are faked; the Ollama fake has ollama.chat's real signature.
"""

import asyncio
import subprocess
import urllib.error
from types import SimpleNamespace

import httpx
import pytest

import merceka_core.llm as llm_module
from merceka_core.llm import LLM


@pytest.fixture(autouse=True)
def _no_verify(monkeypatch):
  monkeypatch.setattr(LLM, "_verify", lambda self: None)


@pytest.fixture
def ollama_calls(monkeypatch):
  """Record ollama_chat calls; unknown kwargs raise TypeError like ollama.chat."""
  import inspect

  import ollama

  calls = []
  signature = inspect.signature(ollama.chat)

  def fake_chat(*args, **kwargs):
    bound = signature.bind(*args, **kwargs)  # TypeError on kwargs ollama.chat rejects
    calls.append({"model": bound.arguments["model"], "options": bound.arguments.get("options")})
    return SimpleNamespace(message=SimpleNamespace(content="local answer", tool_calls=None))

  monkeypatch.setattr(llm_module, "ollama_chat", fake_chat)
  return calls


def _failing(error):
  def transport(self, *_args, **_kwargs):
    raise error
  return transport


def lookup(key: str) -> str:
  """Look up a value.

  Args:
    key: What to look up.
  """
  TOOL_RUNS.append(key)
  return f"value_for_{key}"


TOOL_RUNS: list[str] = []


@pytest.fixture(autouse=True)
def _reset_tool_runs():
  TOOL_RUNS.clear()


_TOOL_CALL = {
  "role": "assistant",
  "content": None,
  "tool_calls": [
    {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": {"key": "x"}}},
  ],
}


class TestFallbackRunsOnce:
  def test_cli_tools_fallback_transport_error_does_not_rerun_the_tool_loop(self, monkeypatch):
    """Regression (review R4): the fallback served the call, failed mid-loop,
    and generate()'s cascade ran the whole tool loop (and its tools) again."""
    raw_calls = []

    def raw(self, *_args, **_kwargs):
      raw_calls.append(self.model_name)
      if len(raw_calls) == 1:
        return dict(_TOOL_CALL)
      raise urllib.error.URLError("connection reset")

    monkeypatch.setattr(LLM, "_cloud_call_raw", raw)
    llm = LLM("claude/sonnet", tools=[lookup], fallback="openrouter/fb")
    with pytest.raises(urllib.error.URLError):
      llm.generate("pay")
    assert raw_calls == ["openrouter/fb", "openrouter/fb"]
    assert TOOL_RUNS == ["x"]

  def test_async_cli_tools_fallback_transport_error_does_not_rerun(self, monkeypatch):
    raw_calls = []

    async def raw(self, *_args, **_kwargs):
      raw_calls.append(self.model_name)
      if len(raw_calls) == 1:
        return dict(_TOOL_CALL)
      raise httpx.ConnectError("down")

    monkeypatch.setattr(LLM, "_acloud_call_raw", raw)
    llm = LLM("codex/gpt-5", tools=[lookup], fallback="openrouter/fb")
    with pytest.raises(httpx.ConnectError):
      asyncio.run(llm.agenerate("pay"))
    assert raw_calls == ["openrouter/fb", "openrouter/fb"]
    assert TOOL_RUNS == ["x"]

  def test_primary_failure_runs_the_fallback_once(self, monkeypatch, ollama_calls):
    monkeypatch.setattr(LLM, "_cloud_call", _failing(ConnectionError("down")))
    assert LLM("openrouter/x", fallback="gemma4:26b").generate("q") == "local answer"
    assert len(ollama_calls) == 1


class TestFallbackKwargs:
  def test_openrouter_sampling_kwargs_become_ollama_options(self, monkeypatch, ollama_calls):
    """Regression (review R5a): temperature/max_tokens made the Ollama fallback
    raise TypeError, masking the primary's error."""
    monkeypatch.setattr(LLM, "_cloud_call", _failing(ConnectionError("down")))
    llm = LLM("openrouter/x", fallback="gemma4:26b")
    out = llm.generate(
      "q", temperature=0.0, max_tokens=300, http_referer="http://x", x_title="t",
      provider={"order": ["a"]},
    )
    assert out == "local answer"
    assert ollama_calls[0]["options"] == {"temperature": 0.0, "num_predict": 300}

  def test_cli_timeout_is_not_passed_to_an_ollama_fallback(self, monkeypatch, ollama_calls):
    monkeypatch.setattr(
      llm_module.subprocess, "run", _failing(subprocess.TimeoutExpired("claude", 30)).__get__(0))
    llm = LLM("claude/sonnet", fallback="gemma4:26b")
    assert llm.generate("q", timeout=30) == "local answer"

  @pytest.mark.asyncio
  async def test_async_cli_timeout_is_not_passed_to_an_ollama_fallback(
    self, monkeypatch, ollama_calls,
  ):
    monkeypatch.setattr(LLM, "_claude_call", _failing(FileNotFoundError("claude")))
    llm = LLM("claude/sonnet", fallback="gemma4:26b")
    assert await llm.agenerate("q", timeout=30) == "local answer"

  def test_tools_fallback_loop_on_ollama_gets_no_cli_timeout(self, ollama_calls):
    llm = LLM("claude/sonnet", tools=[lookup], fallback="gemma4:26b")
    assert llm.generate("q", timeout=30) == "local answer"
    assert ollama_calls[0]["model"] == "gemma4:26b"

  def test_stream_fallback_gets_adapted_kwargs(self, monkeypatch, ollama_calls):
    def failing_stream(self, *_args, **_kwargs):
      yield from ()
      raise FileNotFoundError("claude")

    monkeypatch.setattr(LLM, "_claude_stream", failing_stream)
    llm = LLM("claude/sonnet", fallback="gemma4:26b")
    assert list(llm.stream_generate("q", timeout=30, temperature=0.2)) == ["local answer"]
    assert ollama_calls[0]["options"] == {"temperature": 0.2}

  def test_claude_stream_with_tools_still_falls_back(self, monkeypatch):
    """The Claude stream ignores Python tools, so its failure is a primary
    failure even when tools would route generate() to the fallback."""
    def failing_stream(self, *_args, **_kwargs):
      yield from ()
      raise FileNotFoundError("claude")

    monkeypatch.setattr(LLM, "_claude_stream", failing_stream)
    monkeypatch.setattr(LLM, "_run_tool_loop", lambda self, *_a, **_k: (self.model_name, []))
    llm = LLM("claude/sonnet", tools=[lookup], fallback="openrouter/fb")
    assert list(llm.stream_generate("q")) == ["openrouter/fb"]

  def test_openrouter_fallback_keeps_openrouter_kwargs(self, monkeypatch):
    seen = {}

    def cloud(self, *_args, **kwargs):
      seen[self.model_name] = kwargs
      if self.model_name == "openrouter/primary":
        raise ConnectionError("down")
      return "ok"

    monkeypatch.setattr(LLM, "_cloud_call", cloud)
    llm = LLM("openrouter/primary", fallback="openrouter/fb")
    kwargs = {"temperature": 0.1, "logprobs": True, "top_logprobs": 2, "provider": {"order": ["a"]}}
    assert llm.generate("q", keep_alive="5m", **kwargs) == "ok"
    assert seen["openrouter/fb"] == kwargs

  def test_cli_timeout_carries_over_to_an_openrouter_fallback(self, monkeypatch):
    """OpenRouter uses timeout= as its HTTP timeout, so the fallback keeps it."""
    seen = {}

    def cloud(self, *_args, **kwargs):
      seen.update(kwargs)
      return "ok"

    monkeypatch.setattr(LLM, "_claude_call", _failing(FileNotFoundError("claude")))
    monkeypatch.setattr(LLM, "_cloud_call", cloud)
    assert LLM("claude/sonnet", fallback="openrouter/fb").generate("q", timeout=600) == "ok"
    assert seen == {"timeout": 600}

  def test_codex_images_keep_a_codex_fallback(self, monkeypatch, tmp_path):
    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG")
    runs = []

    def fake_run(cmd, **_kwargs):
      runs.append(cmd)
      failed = len(runs) == 1
      return SimpleNamespace(returncode=1 if failed else 0, stdout="seen", stderr="")

    monkeypatch.setattr(llm_module.subprocess, "run", fake_run)
    llm = LLM("codex/gpt-5", fallback="codex/default")
    assert llm.generate("describe", images=[str(png)]) == "seen"
    assert all(str(png) in cmd for cmd in runs) and len(runs) == 2

  def test_codex_images_are_not_answered_blind_by_another_backend(self, monkeypatch, tmp_path):
    """Regression (review R15): images= went into the OpenRouter body and the
    fallback answered without seeing the image."""
    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG")
    monkeypatch.setattr(
      llm_module.subprocess, "run",
      lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="not logged in"))
    monkeypatch.setattr(LLM, "_cloud_call", lambda *_a, **_k: pytest.fail("fallback ran blind"))
    llm = LLM("codex/default", fallback="openrouter/google/gemini-3-flash")
    with pytest.raises(subprocess.CalledProcessError):
      llm.generate("describe this", images=[str(png)])

  def test_tools_fallback_refuses_images_it_cannot_see(self, monkeypatch, tmp_path):
    monkeypatch.setattr(LLM, "_run_tool_loop", lambda *_a, **_k: pytest.fail("ran blind"))
    llm = LLM("codex/gpt-5", tools=[lookup], fallback="openrouter/fb")
    with pytest.raises(ValueError, match="images"):
      llm.generate("describe", images=[str(tmp_path / "x.png")])


class TestCascadeCatchesProviderFailures:
  def test_ollama_response_error_falls_back(self, monkeypatch):
    """Regression (review R12): an Ollama server error never reached the fallback."""
    import ollama

    def ollama_500(**_kwargs):
      raise ollama.ResponseError("model requires more system memory", 500)

    monkeypatch.setattr(llm_module, "ollama_chat", ollama_500)
    monkeypatch.setattr(LLM, "_cloud_call", lambda self, *_a, **_k: f"from {self.model_name}")
    assert LLM("gemma4:26b", fallback="openrouter/x").generate("q") == "from openrouter/x"

  @pytest.mark.asyncio
  async def test_async_ollama_response_error_falls_back(self, monkeypatch):
    import ollama

    def ollama_500(**_kwargs):
      raise ollama.ResponseError("model requires more system memory", 500)

    async def cloud(self, *_args, **_kwargs):
      return f"from {self.model_name}"

    monkeypatch.setattr(llm_module, "ollama_chat", ollama_500)
    monkeypatch.setattr(LLM, "_acloud_call", cloud)
    assert await LLM("gemma4:26b", fallback="openrouter/x").agenerate("q") == "from openrouter/x"

  def test_ollama_response_error_without_fallback_raises(self, monkeypatch):
    import ollama

    def ollama_500(**_kwargs):
      raise ollama.ResponseError("boom", 500)

    monkeypatch.setattr(llm_module, "ollama_chat", ollama_500)
    with pytest.raises(ollama.ResponseError):
      LLM("gemma4:26b").generate("q")


class _Urlopen:
  """urlopen stand-in returning ``bodies`` in order (JSON-encoded)."""

  def __init__(self, bodies):
    self.bodies = list(bodies)
    self.models = []

  def __call__(self, request, timeout=None):
    import json

    del timeout
    self.models.append(json.loads(request.data)["model"])
    body = json.dumps(self.bodies.pop(0)).encode()

    class Response:
      def __enter__(self):
        return self

      def __exit__(self, *exc):
        return False

      def read(self, *_args):
        return body

    return Response()


_ERROR_BODY = {"error": {"code": 502, "message": "Provider returned error: upstream overloaded"}}
_NO_CONTENT_BODY = {
  "id": "gen-empty",
  "choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "length"}],
  "usage": {"cost": 0.01},
}
_OK_BODY = {"choices": [{"message": {"role": "assistant", "content": "fallback answer"}}]}


class TestProviderResponseErrors:
  @pytest.fixture(autouse=True)
  def _key(self, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

  def test_error_body_raises_a_typed_error(self, monkeypatch):
    """Regression (review R10): KeyError('choices'), outside the cascade."""
    from merceka_core.errors import LLMResponseError

    monkeypatch.setattr(llm_module, "urlopen", _Urlopen([_ERROR_BODY]))
    with pytest.raises(LLMResponseError, match="upstream overloaded"):
      LLM("openrouter/x").generate("q")

  def test_error_body_falls_back_without_retrying(self, monkeypatch):
    fake = _Urlopen([_ERROR_BODY, _OK_BODY])
    monkeypatch.setattr(llm_module, "urlopen", fake)
    assert LLM("openrouter/x", fallback="openrouter/y").generate("q") == "fallback answer"
    assert fake.models == ["x", "y"]  # the processed request is not sent again

  def test_missing_content_raises_with_the_finish_reason(self, monkeypatch):
    """Regression (review R13): a bare assert, skipped under python -O."""
    from merceka_core.errors import LLMResponseError

    monkeypatch.setattr(llm_module, "urlopen", _Urlopen([_NO_CONTENT_BODY]))
    with pytest.raises(LLMResponseError, match="finish_reason='length'"):
      LLM("openrouter/x").generate("q", max_tokens=400)

  def test_missing_content_falls_back(self, monkeypatch):
    monkeypatch.setattr(llm_module, "urlopen", _Urlopen([_NO_CONTENT_BODY, _OK_BODY]))
    assert LLM("openrouter/x", fallback="openrouter/y").generate("q") == "fallback answer"

  def test_async_error_body_falls_back(self, monkeypatch):
    bodies = [_ERROR_BODY, _OK_BODY]

    class FakeResponse:
      def __init__(self, body):
        self._body = body

      def raise_for_status(self):
        return None

      def json(self):
        return self._body

    class FakeAsyncClient:
      def __init__(self, *_args, **_kwargs):
        pass

      async def __aenter__(self):
        return self

      async def __aexit__(self, *exc):
        return False

      async def post(self, *_args, **_kwargs):
        return FakeResponse(bodies.pop(0))

    monkeypatch.setattr(llm_module.httpx, "AsyncClient", FakeAsyncClient)
    llm = LLM("openrouter/x", fallback="openrouter/y")
    assert asyncio.run(llm.agenerate("q")) == "fallback answer"

  def test_tool_loop_error_body_raises_a_typed_error(self, monkeypatch):
    from merceka_core.errors import LLMResponseError

    monkeypatch.setattr(llm_module, "urlopen", _Urlopen([_ERROR_BODY]))
    with pytest.raises(LLMResponseError):
      LLM("openrouter/x", tools=[lookup]).generate("q")

  def test_ollama_missing_content_raises_a_typed_error(self, monkeypatch):
    from merceka_core.errors import LLMResponseError

    monkeypatch.setattr(
      llm_module, "ollama_chat",
      lambda **_kwargs: SimpleNamespace(message=SimpleNamespace(content=None)))
    with pytest.raises(LLMResponseError, match="no content"):
      LLM("gemma4:26b").generate("q")

  def test_error_body_is_still_metered(self, monkeypatch):
    """Recorded before parsing: whether OpenRouter billed it is unknown offline."""
    from merceka_core import costs

    monkeypatch.setattr(llm_module, "urlopen", _Urlopen([_NO_CONTENT_BODY]))
    with pytest.raises(Exception):
      LLM("openrouter/x").generate("q")
    rows = costs.ledger_path().read_text().splitlines()
    assert len(rows) == 1 and '"usd": 0.01' in rows[0]


class TestAsyncFallbackConstruction:
  """Building a local fallback runs _verify, a blocking Ollama HTTP call (and a
  model pull when missing). Regression (review R11): it ran on the event loop."""

  @pytest.fixture
  def verify_threads(self, monkeypatch):
    import threading

    seen = {}
    monkeypatch.setattr(
      LLM, "_verify", lambda self: seen.setdefault(self.model_name, threading.get_ident()))
    return seen

  @pytest.mark.asyncio
  async def test_cascade_builds_the_fallback_off_the_event_loop(
    self, monkeypatch, verify_threads, ollama_calls,
  ):
    import threading

    async def failing(self, *_args, **_kwargs):
      raise httpx.ConnectError("down")

    monkeypatch.setattr(LLM, "_acloud_call", failing)
    llm = LLM("openrouter/x", fallback="gemma4:26b")
    assert await llm.agenerate("q") == "local answer"
    assert verify_threads["gemma4:26b"] != threading.get_ident()

  @pytest.mark.asyncio
  async def test_tools_fallback_is_built_off_the_event_loop(
    self, verify_threads, ollama_calls,
  ):
    import threading

    llm = LLM("claude/sonnet", tools=[lookup], fallback="gemma4:26b")
    assert await llm.agenerate("q") == "local answer"
    assert verify_threads["gemma4:26b"] != threading.get_ident()
