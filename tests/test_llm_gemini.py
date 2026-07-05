"""Tests for llm_gemini.py: upload polling, video config translation, the
video call happy/retry/cleanup paths, and grounding metadata extraction.

Everything runs offline: the google-genai client factory is monkeypatched
with hand-rolled fakes, and time.sleep/time.monotonic are replaced with a
fake clock so no test ever actually waits.
"""

from types import SimpleNamespace

import pytest

from merceka_core import llm_gemini
from merceka_core.errors import (
  VideoBackendError,
  VideoNotFoundError,
  VideoUploadError,
)
from merceka_core.llm_gemini import (
  _build_video_config,
  _extract_grounding,
  _gemini_poll_until_active,
  _gemini_video_call,
)


def _file_obj(name="files/vid1", state="PROCESSING"):
  return SimpleNamespace(name=name, state=SimpleNamespace(name=state))


class FakeClock:
  """Deterministic replacement for time.monotonic/time.sleep."""

  def __init__(self):
    self.now = 0.0
    self.sleeps = []

  def monotonic(self):
    return self.now

  def sleep(self, seconds):
    self.sleeps.append(seconds)
    self.now += seconds


@pytest.fixture
def clock(monkeypatch):
  c = FakeClock()
  monkeypatch.setattr(llm_gemini.time, "monotonic", c.monotonic)
  monkeypatch.setattr(llm_gemini.time, "sleep", c.sleep)
  return c


class FakeFiles:
  def __init__(self, get_states=None):
    self.get_states = list(get_states or [])
    self.get_calls = []
    self.deleted = []

  def upload(self, file):
    return _file_obj(name=f"files/{file}", state="ACTIVE")

  def get(self, name):
    self.get_calls.append(name)
    state = self.get_states.pop(0) if self.get_states else "PROCESSING"
    return _file_obj(name=name, state=state)

  def delete(self, name):
    self.deleted.append(name)


class FakeModels:
  def __init__(self, outcomes):
    """outcomes: list of Exception (raised) or response objects (returned)."""
    self.outcomes = list(outcomes)
    self.calls = []

  def generate_content(self, **kwargs):
    self.calls.append(kwargs)
    outcome = self.outcomes.pop(0)
    if isinstance(outcome, Exception):
      raise outcome
    return outcome


class FakeClient:
  def __init__(self, files=None, models=None):
    self.files = files or FakeFiles()
    self.models = models or FakeModels([SimpleNamespace(text="ok")])


class FakeLLM:
  model_name = "gemini/gemini-2.5-pro"
  system_prompt = ""

  def _parse_response(self, text):
    return f"parsed:{text}"


# --- _gemini_poll_until_active ---

class TestPollUntilActive:
  def test_active_immediately_returns_without_polling(self, clock):
    files = FakeFiles()
    client = FakeClient(files=files)
    file_obj = _file_obj(state="ACTIVE")

    result = _gemini_poll_until_active(client, file_obj, timeout_s=10, poll_interval_s=1)

    assert result is file_obj
    assert files.get_calls == []
    assert clock.sleeps == []

  def test_failed_state_raises_upload_error(self, clock):
    client = FakeClient()
    file_obj = _file_obj(name="files/broken", state="FAILED")

    with pytest.raises(VideoUploadError, match="FAILED.*files/broken"):
      _gemini_poll_until_active(client, file_obj, timeout_s=10, poll_interval_s=1)

  def test_polls_files_get_until_active(self, clock):
    files = FakeFiles(get_states=["PROCESSING", "ACTIVE"])
    client = FakeClient(files=files)
    file_obj = _file_obj(name="files/vid1")

    result = _gemini_poll_until_active(client, file_obj, timeout_s=100, poll_interval_s=2)

    assert result.state.name == "ACTIVE"
    assert files.get_calls == ["files/vid1", "files/vid1"]
    assert clock.sleeps == [2, 2]

  def test_timeout_raises_with_state_in_message(self, clock):
    files = FakeFiles()  # stays PROCESSING forever
    client = FakeClient(files=files)
    file_obj = _file_obj(name="files/slow")

    with pytest.raises(VideoUploadError, match="within 5.*state=PROCESSING"):
      _gemini_poll_until_active(client, file_obj, timeout_s=5, poll_interval_s=1)


# --- _build_video_config ---

class TestBuildVideoConfig:
  def test_max_tokens_becomes_max_output_tokens(self):
    config, extra = _build_video_config(max_tokens=512)
    assert config.max_output_tokens == 512
    assert extra == {}

  def test_system_prompt_becomes_system_instruction(self):
    config, extra = _build_video_config(system_prompt="be brief")
    assert config.system_instruction == "be brief"

  def test_known_config_keys_moved_out_of_extra(self):
    config, extra = _build_video_config(
      max_tokens=100, temperature=0.2, top_p=0.9, custom_flag="stays",
    )
    assert config.temperature == 0.2
    assert config.top_p == 0.9
    assert extra == {"custom_flag": "stays"}

  def test_no_config_fields_returns_none(self):
    config, extra = _build_video_config(passthrough=1)
    assert config is None
    assert extra == {"passthrough": 1}


# --- _gemini_video_call ---

@pytest.fixture
def video(tmp_path):
  path = tmp_path / "clip.mp4"
  path.write_bytes(b"not really a video")
  return path


class TestGeminiVideoCall:
  def test_happy_path_uploads_generates_parses_deletes(self, clock, monkeypatch, video):
    models = FakeModels([SimpleNamespace(text="the answer")])
    client = FakeClient(models=models)
    monkeypatch.setattr(llm_gemini, "_gemini_client", lambda: client)

    result = _gemini_video_call(
      FakeLLM(), "what happens?", video, timeout_s=10, poll_interval_s=1,
    )

    assert result == "parsed:the answer"
    call = models.calls[0]
    assert call["model"] == "gemini-2.5-pro"  # gemini/ prefix stripped
    assert call["contents"][-1] == "what happens?"
    assert client.files.deleted == [f"files/{video}"]

  def test_missing_video_raises_eagerly_before_client(self, monkeypatch, tmp_path):
    def no_client():
      raise AssertionError("client must not be constructed for a missing file")

    monkeypatch.setattr(llm_gemini, "_gemini_client", no_client)

    with pytest.raises(VideoNotFoundError, match="nope.mp4"):
      _gemini_video_call(
        FakeLLM(), "hi", tmp_path / "nope.mp4", timeout_s=10, poll_interval_s=1,
      )

  def test_retryable_5xx_then_success(self, clock, monkeypatch, video):
    transient = Exception("503 unavailable")
    transient.status_code = 503
    models = FakeModels([transient, SimpleNamespace(text="recovered")])
    client = FakeClient(models=models)
    monkeypatch.setattr(llm_gemini, "_gemini_client", lambda: client)

    result = _gemini_video_call(
      FakeLLM(), "hi", video, timeout_s=10, poll_interval_s=1,
    )

    assert result == "parsed:recovered"
    assert len(models.calls) == 2
    assert len(clock.sleeps) == 1  # backoff went through the fake clock

  def test_non_retryable_error_raises_backend_error_and_still_deletes(
    self, clock, monkeypatch, video,
  ):
    fatal = Exception("400 bad request")
    fatal.status_code = 400
    client = FakeClient(models=FakeModels([fatal]))
    monkeypatch.setattr(llm_gemini, "_gemini_client", lambda: client)

    with pytest.raises(VideoBackendError, match="generate_content failed"):
      _gemini_video_call(FakeLLM(), "hi", video, timeout_s=10, poll_interval_s=1)

    assert client.files.deleted == [f"files/{video}"]
    assert clock.sleeps == []

  def test_retries_exhausted_raises_backend_error(self, clock, monkeypatch, video):
    def transient():
      exc = Exception("503")
      exc.status_code = 503
      return exc

    client = FakeClient(models=FakeModels([transient(), transient(), transient()]))
    monkeypatch.setattr(llm_gemini, "_gemini_client", lambda: client)

    with pytest.raises(VideoBackendError):
      _gemini_video_call(FakeLLM(), "hi", video, timeout_s=10, poll_interval_s=1)

    assert len(clock.sleeps) == 2  # 3 attempts, 2 backoffs
    assert client.files.deleted == [f"files/{video}"]


# --- _extract_grounding ---

class TestExtractGrounding:
  def test_full_metadata(self):
    chunk = SimpleNamespace(web=SimpleNamespace(uri="https://x.test", title="X"))
    gm = SimpleNamespace(
      web_search_queries=["q1", "q2"],
      grounding_chunks=[chunk, SimpleNamespace(web=None)],
      search_entry_point=SimpleNamespace(rendered_content="<div>chip</div>"),
    )
    response = SimpleNamespace(candidates=[SimpleNamespace(grounding_metadata=gm)])

    out = _extract_grounding(response)

    assert out["queries"] == ["q1", "q2"]
    assert out["citations"] == [{"uri": "https://x.test", "title": "X"}]
    assert out["search_entry_point_html"] == "<div>chip</div>"

  def test_no_candidates_returns_empty(self):
    out = _extract_grounding(SimpleNamespace(candidates=[]))
    assert out == {"queries": [], "citations": [], "search_entry_point_html": None}

  def test_absent_grounding_metadata_returns_empty(self):
    # python-genai #802: metadata may be missing even when searches ran.
    response = SimpleNamespace(candidates=[SimpleNamespace(grounding_metadata=None)])
    out = _extract_grounding(response)
    assert out == {"queries": [], "citations": [], "search_entry_point_html": None}
