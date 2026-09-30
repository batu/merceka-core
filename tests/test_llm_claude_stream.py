"""LLM._claude_stream / stream_generate: CLI failures surface, children are reaped.

subprocess.Popen is replaced by a fake whose wait() refuses to block forever on a
live child, so an unbounded wait fails the test instead of hanging the suite.
"""

import itertools
import json
import os
import subprocess
import time

import pytest

import merceka_core.llm as llm_module
from merceka_core.llm import LLM


def _delta(text):
  return json.dumps({
    "type": "stream_event",
    "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}},
  }) + "\n"


def _result(is_error=False, text="done"):
  return json.dumps({
    "type": "result",
    "subtype": "success",
    "is_error": is_error,
    "result": text,
  }) + "\n"


class _Stdin:
  def __init__(self, broken=False):
    self.broken = broken
    self.written = ""
    self.closed = False

  def write(self, data):
    if self.broken:
      raise BrokenPipeError(32, "Broken pipe")
    self.written += data

  def close(self):
    self.closed = True


class _Pipe:
  def __init__(self, lines):
    self._lines = iter(lines)
    self.closed = False

  def __iter__(self):
    return self

  def __next__(self):
    return next(self._lines)

  def close(self):
    self.closed = True


def _stderr_pipe(text, *, hold_open=False):
  """A real pipe holding ``text``, as the child's stderr.

  ``hold_open`` keeps the write end open, like a grandchild that inherited it.
  Returns the read end and, when held open, the write fd to close later.
  """
  read_fd, write_fd = os.pipe()
  os.write(write_fd, text.encode())
  if not hold_open:
    os.close(write_fd)
    write_fd = None
  return os.fdopen(read_fd, "r"), write_fd


class FakePopen:
  """A Claude CLI child.

  ``exits`` is the code the child exits with once its output is drained;
  ``linger`` keeps it alive after that until it is signalled; ``stubborn``
  makes it ignore SIGTERM.
  """

  def __init__(self, lines, *, exits=0, stderr="", linger=False, stubborn=False,
               broken_stdin=False, hold_stderr=False):
    self.stdin = _Stdin(broken=broken_stdin)
    self.stdout = _Pipe(lines)
    self.stderr, self.stderr_writer = _stderr_pipe(stderr, hold_open=hold_stderr)
    self._exits = exits
    self._linger = linger
    self._stubborn = stubborn
    self.returncode = None
    self.signals = []
    self.unbounded_waits = 0

  def _alive(self):
    if self.returncode is None and not self._linger and not self.signals:
      self.returncode = self._exits  # finished on its own
    return self.returncode is None

  def poll(self):
    return None if self._alive() else self.returncode

  def wait(self, timeout=None):
    if self._alive():
      if timeout is None:
        self.unbounded_waits += 1
        raise AssertionError("wait() without a timeout on a live child")
      raise subprocess.TimeoutExpired("claude", timeout)
    return self.returncode

  def terminate(self):
    self.signals.append("TERM")
    if not self._stubborn and self.returncode is None:
      self.returncode = -15

  def kill(self):
    self.signals.append("KILL")
    if self.returncode is None:
      self.returncode = -9


@pytest.fixture
def popen(monkeypatch):
  """Install one FakePopen per test; returns a setter taking FakePopen kwargs."""
  holder = {}

  def install(lines, **kwargs):
    def factory(cmd, **_popen_kwargs):
      holder["proc"] = FakePopen(lines, **kwargs)
      holder["cmd"] = cmd
      return holder["proc"]

    monkeypatch.setattr(llm_module.subprocess, "Popen", factory)
    return holder

  return install


@pytest.fixture
def fallback_answer(monkeypatch):
  calls = []

  def cloud(self, *_args, **_kwargs):
    calls.append(self.model_name)
    return "fallback-answer"

  monkeypatch.setattr(LLM, "_cloud_call", cloud)
  return calls


class TestClaudeStreamFailures:
  def test_happy_path_yields_deltas_and_reaps_without_signals(self, popen):
    holder = popen([_delta("Hel"), _delta("lo"), _result()])
    assert list(LLM("claude/sonnet").stream_generate("q")) == ["Hel", "lo"]
    proc = holder["proc"]
    assert proc.returncode == 0 and proc.signals == []
    assert proc.stdout.closed and proc.stderr.closed and proc.stdin.closed

  def test_error_result_event_raises_with_detail(self, popen):
    popen([_result(is_error=True, text="Claude AI usage limit reached")],
          exits=1, stderr="Error: usage limit\n")
    with pytest.raises(subprocess.CalledProcessError) as info:
      list(LLM("claude/sonnet").stream_generate("q"))
    assert info.value.returncode == 1
    assert "usage limit reached" in info.value.stderr
    assert "Error: usage limit" in info.value.stderr

  def test_error_result_with_zero_exit_still_raises(self, popen):
    popen([_result(is_error=True, text="auth failed")], exits=0)
    with pytest.raises(subprocess.CalledProcessError, match="exit status 1"):
      list(LLM("claude/sonnet").stream_generate("q"))

  def test_nonzero_exit_without_result_event_raises(self, popen):
    popen([], exits=2, stderr="fatal: bad flag\n")
    with pytest.raises(subprocess.CalledProcessError) as info:
      list(LLM("claude/sonnet").stream_generate("q"))
    assert info.value.returncode == 2 and "bad flag" in info.value.stderr

  def test_stderr_held_open_by_a_grandchild_does_not_block(self, popen, monkeypatch):
    monkeypatch.setattr(llm_module, "_STREAM_STDERR_WAIT_S", 0.2)
    holder = popen([], exits=1, stderr="partial detail", hold_stderr=True)
    start = time.monotonic()
    try:
      with pytest.raises(subprocess.CalledProcessError) as info:
        list(LLM("claude/sonnet").stream_generate("q"))
    finally:
      os.close(holder["proc"].stderr_writer)
    assert time.monotonic() - start < 2.0
    assert "partial detail" in info.value.stderr

  def test_error_result_without_fallback_raises(self, popen):
    popen([_result(is_error=True, text="not logged in")], exits=1)
    with pytest.raises(subprocess.CalledProcessError, match="exit status 1"):
      list(LLM("claude/sonnet").stream_generate("q"))

  def test_error_result_falls_back(self, popen, fallback_answer):
    popen([_result(is_error=True, text="usage limit")], exits=1)
    llm = LLM("claude/sonnet", fallback="openrouter/fb")
    assert list(llm.stream_generate("q")) == ["fallback-answer"]
    assert fallback_answer == ["openrouter/fb"]

  def test_failure_after_output_raises_instead_of_appending_a_fallback(
    self, popen, fallback_answer,
  ):
    """Chunks already reached the consumer; a second answer after them is corrupt."""
    popen([_delta("partial"), _result(is_error=True, text="max turns")], exits=1)
    gen = LLM("claude/sonnet", fallback="openrouter/fb").stream_generate("q")
    assert next(gen) == "partial"
    with pytest.raises(subprocess.CalledProcessError):
      next(gen)
    assert fallback_answer == []


class TestClaudeStreamTeardown:
  def test_abandoned_stream_terminates_child(self, popen):
    holder = popen(itertools.repeat(_delta("x")), linger=True)
    gen = LLM("claude/sonnet").stream_generate("q")
    assert next(gen) == "x"
    gen.close()
    proc = holder["proc"]
    assert proc.signals == ["TERM"]
    assert proc.poll() == -15
    assert proc.stdout.closed and proc.stderr.closed

  def test_child_ignoring_sigterm_is_killed(self, popen):
    holder = popen(itertools.repeat(_delta("x")), linger=True, stubborn=True)
    gen = LLM("claude/sonnet").stream_generate("q")
    next(gen)
    gen.close()
    assert holder["proc"].signals == ["TERM", "KILL"]
    assert holder["proc"].poll() == -9

  def test_child_lingering_after_success_result_is_terminated(self, popen):
    holder = popen([_delta("answer"), _result()], linger=True)
    assert list(LLM("claude/sonnet").stream_generate("q")) == ["answer"]
    assert holder["proc"].signals == ["TERM"]
    assert holder["proc"].unbounded_waits == 0

  def test_broken_stdin_reaps_child_and_falls_back(self, popen, fallback_answer):
    holder = popen([], exits=1, broken_stdin=True, linger=True)
    llm = LLM("claude/sonnet", fallback="openrouter/fb")
    assert list(llm.stream_generate("q")) == ["fallback-answer"]
    proc = holder["proc"]
    assert proc.poll() is not None
    assert proc.stdout.closed and proc.stderr.closed


# --- Real child processes (a Python script stands in for the claude binary) ---

_FAILING_CLI = """
import json, sys
sys.stdin.read()
sys.stderr.write("Error: usage limit\\n")
print(json.dumps({"type": "result", "subtype": "success", "is_error": True,
                  "result": "Claude AI usage limit reached"}), flush=True)
sys.exit(1)
"""

_STALLED_CLI = """
import json, sys, time
sys.stdin.read()
event = {"type": "stream_event", "event": {"type": "content_block_delta",
         "delta": {"type": "text_delta", "text": "tick"}}}
print(json.dumps(event), flush=True)
time.sleep(600)  # a stalled CLI: no output, no exit
"""

_SILENT_CLI = """
import sys, time
sys.stdin.read()
time.sleep(600)  # a stalled CLI that never writes anything
"""


@pytest.fixture
def real_cli(monkeypatch):
  import sys

  started = []
  real_popen = subprocess.Popen

  def install(script):
    monkeypatch.setattr(
      llm_module._cli, "claude_command", lambda *_args, **_kwargs: [sys.executable, "-c", script])

    def spy(*args, **kwargs):
      started.append(real_popen(*args, **kwargs))
      return started[-1]

    monkeypatch.setattr(llm_module.subprocess, "Popen", spy)
    return started

  return install


class TestClaudeStreamRealProcess:
  def test_failing_cli_raises_with_stderr(self, real_cli):
    started = real_cli(_FAILING_CLI)
    with pytest.raises(subprocess.CalledProcessError) as info:
      list(LLM("claude/sonnet").stream_generate("q"))
    assert info.value.returncode == 1
    assert "usage limit reached" in info.value.stderr
    assert "Error: usage limit" in info.value.stderr
    assert started[0].poll() == 1

  def test_abandoned_stalled_stream_terminates_the_child(self, real_cli):
    import threading

    started = real_cli(_STALLED_CLI)
    gen = LLM("claude/sonnet").stream_generate("q")
    assert next(gen) == "tick"
    closer = threading.Thread(target=gen.close, daemon=True)
    closer.start()
    closer.join(timeout=10)
    try:
      assert not closer.is_alive(), "stream teardown blocked on a stalled child"
      assert started[0].poll() is not None  # reaped, not left running
    finally:
      if started[0].poll() is None:
        started[0].kill()
        closer.join(timeout=10)

  def test_abandoned_astream_terminates_a_stalled_child(self, real_cli):
    """Card 03 B: the producer thread is blocked reading a stalled CLI, so the
    consumer's teardown (aclose) waited on the child indefinitely."""
    import asyncio

    started = real_cli(_STALLED_CLI)

    async def consume():
      agen = LLM("claude/sonnet").astream_generate("q")
      async for chunk in agen:
        assert chunk == "tick"
        break
      await agen.aclose()

    async def main():
      try:
        await asyncio.wait_for(consume(), timeout=5)
      finally:
        if started and started[0].poll() is None:
          started[0].kill()  # unblock the producer thread if teardown failed

    start = time.monotonic()
    asyncio.run(main())
    assert time.monotonic() - start < 5
    assert started[0].returncode == -15  # terminated by the teardown, not killed above

  def test_astream_cancelled_before_output_does_not_start_the_fallback(
    self, real_cli, fallback_answer,
  ):
    import asyncio

    started = real_cli(_SILENT_CLI)

    async def main():
      agen = LLM("claude/sonnet", fallback="openrouter/fb").astream_generate("q")
      try:
        with pytest.raises(asyncio.TimeoutError):
          await asyncio.wait_for(agen.__anext__(), timeout=0.5)
      finally:
        if started and started[0].poll() is None:
          started[0].kill()

    asyncio.run(asyncio.wait_for(main(), timeout=10))
    assert started[0].returncode == -15
    assert fallback_answer == []

  def test_astream_producer_keeps_cost_attribution(self, monkeypatch):
    """The producer thread runs in a copy of the caller's context, so
    costs.attribution() also covers spend inside a streamed fallback."""
    import asyncio

    from merceka_core import costs

    seen = []

    def stream(self, *_args, **_kwargs):
      seen.append(costs._attribution_var.get())
      yield "x"

    monkeypatch.setattr(LLM, "stream_generate", stream)

    async def main():
      with costs.attribution({"session": "s1"}):
        return [c async for c in LLM("claude/sonnet").astream_generate("q")]

    assert asyncio.run(main()) == ["x"]
    assert seen == [{"session": "s1"}]
