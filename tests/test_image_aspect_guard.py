"""Aspect guard on the edit paths that used to stretch silently.

Every path that resizes a model output back to the input size must refuse an
aspect change beyond the 2% tolerance instead of warping the content. httpx is
faked; each fake answers with an image of a chosen size.
"""

import base64
import io
import json

import pytest
from PIL import Image

from merceka_core import image as image_module
from merceka_core.image import edit_image, inpaint

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image-preview:generateContent"
OPENROUTER_CHAT = "https://openrouter.ai/api/v1/chat/completions"
FAL_RESULT = "https://fal.media/files/result.png"


def _png(size: tuple[int, int]) -> bytes:
  buf = io.BytesIO()
  Image.new("RGB", size, (0, 90, 200)).save(buf, format="PNG")
  return buf.getvalue()


class FakeResponse:
  def __init__(self, body=None, content=b""):
    self.status_code = 200
    self._body = body or {}
    self.content = content
    self.headers = {}
    self.text = json.dumps(self._body)[:200]

  def json(self):
    return self._body

  def raise_for_status(self):
    return None


class FakeProvider:
  """Serves a POST body built by ``respond(url, json)`` and GETs of the fal result."""

  def __init__(self, respond, result_size=None):
    self.respond = respond
    self.result_size = result_size
    self.posts: list[tuple[str, dict]] = []

  def factory(self, *_args, **_kwargs):
    return self

  def __enter__(self):
    return self

  def __exit__(self, *_exc):
    return False

  def post(self, url, **kwargs):
    self.posts.append((url, kwargs))
    return FakeResponse(body=self.respond(url, kwargs.get("json")))

  def get(self, url, **_kwargs):
    assert url == FAL_RESULT and self.result_size is not None
    return FakeResponse(content=_png(self.result_size))


@pytest.fixture
def provider(monkeypatch):
  monkeypatch.delenv("MERCEKA_FORCE_OPENROUTER", raising=False)

  def install(respond, result_size=None) -> FakeProvider:
    fake = FakeProvider(respond, result_size)
    monkeypatch.setattr(image_module.httpx, "Client", fake.factory)
    return fake

  return install


def _gemini_body(size: tuple[int, int]):
  def respond(_url, _json):
    data = base64.b64encode(_png(size)).decode("ascii")
    return {"candidates": [{"content": {"parts": [{"inlineData": {"data": data}}]}}]}

  return respond


def _openrouter_body(size: tuple[int, int]):
  def respond(_url, _json):
    uri = "data:image/png;base64," + base64.b64encode(_png(size)).decode("ascii")
    return {"choices": [{"message": {"images": [{"image_url": {"url": uri}}]}}]}

  return respond


# --- fal inpaint ---


def test_fal_inpaint_refuses_to_stretch_a_different_aspect(provider, monkeypatch):
  monkeypatch.setenv("FAL_KEY", "test-key")
  provider(lambda _url, _json: {"images": [{"url": FAL_RESULT}]}, result_size=(200, 200))

  with pytest.raises(RuntimeError, match="refusing to stretch"):
    inpaint(Image.new("RGB", (300, 200)), Image.new("L", (300, 200), 255), "p")


def test_fal_inpaint_resizes_a_same_aspect_result(provider, monkeypatch):
  monkeypatch.setenv("FAL_KEY", "test-key")
  provider(lambda _url, _json: {"images": [{"url": FAL_RESULT}]}, result_size=(600, 400))

  result = inpaint(Image.new("RGB", (300, 200)), Image.new("L", (300, 200), 255), "p")

  assert result.size == (300, 200)


# --- Google direct edits ---


@pytest.fixture
def google_key(monkeypatch):
  monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


@pytest.mark.usefixtures("google_key")
def test_google_direct_edit_requests_the_input_aspect_not_square(provider):
  fake = provider(_gemini_body((1344, 768)))

  result = edit_image(Image.new("RGB", (1600, 900)), "p", model="google/gemini-3.1-flash-image-preview")

  url, kwargs = fake.posts[0]
  assert url == GEMINI_URL
  assert kwargs["json"]["generationConfig"]["imageConfig"]["aspectRatio"] == "16:9"
  assert result.size == (1600, 900)


@pytest.mark.usefixtures("google_key")
def test_google_direct_edit_refuses_to_stretch_a_different_aspect(provider):
  provider(_gemini_body((1024, 1024)))

  with pytest.raises(RuntimeError, match="refusing to stretch"):
    edit_image(Image.new("RGB", (1600, 900)), "p", model="google/gemini-3.1-flash-image-preview")


@pytest.mark.usefixtures("google_key")
def test_google_direct_edit_without_resize_returns_the_model_size(provider):
  provider(_gemini_body((1024, 1024)))

  result = edit_image(
    Image.new("RGB", (1600, 900)), "p",
    model="google/gemini-3.1-flash-image-preview", resize_to_input=False,
  )

  assert result.size == (1024, 1024)


# --- OpenRouter edits ---


@pytest.fixture
def openrouter_key(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.mark.usefixtures("openrouter_key")
def test_openrouter_edit_refuses_to_stretch_a_different_aspect(provider):
  # The review's repro: a 1100x1000 input came back 1184x864 and was stretched.
  provider(_openrouter_body((1184, 864)))

  with pytest.raises(RuntimeError, match="refusing to stretch"):
    edit_image(Image.new("RGB", (1100, 1000)), "p", model="google/gemini-3.1-flash-image-preview")


@pytest.mark.usefixtures("openrouter_key")
def test_openrouter_edit_requests_the_nearest_supported_aspect(provider):
  fake = provider(_openrouter_body((1248, 832)))

  result = edit_image(Image.new("RGB", (1500, 1000)), "p", model="google/gemini-3.1-flash-image-preview")

  assert fake.posts[0][0] == OPENROUTER_CHAT
  assert fake.posts[0][1]["json"]["image_config"]["aspect_ratio"] == "3:2"
  assert result.size == (1500, 1000)
