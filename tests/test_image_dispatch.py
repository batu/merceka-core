"""Dispatch tests for image.py: which transport generate_image/edit_image
pick per model string (openai/ prefix → OpenAI API, anything else →
OpenRouter chat-completions), and error propagation on failure responses.

httpx.Client is the boundary the code calls, so it is replaced with a
hand-rolled fake that records requests and serves canned responses.
"""

import base64
import io

import pytest
from PIL import Image

from merceka_core import image as image_module
from merceka_core.image import edit_image, generate_image

OPENAI_GENERATIONS = "https://api.openai.com/v1/images/generations"
OPENAI_EDITS = "https://api.openai.com/v1/images/edits"
OPENROUTER_CHAT = "https://openrouter.ai/api/v1/chat/completions"


def _png_b64(size=(1, 1)) -> str:
  img = Image.new("RGB", size, (0, 128, 255))
  buf = io.BytesIO()
  img.save(buf, format="PNG")
  return base64.b64encode(buf.getvalue()).decode("ascii")


def _openai_body():
  return {"data": [{"b64_json": _png_b64()}]}


def _openrouter_body():
  uri = f"data:image/png;base64,{_png_b64()}"
  return {"choices": [{"message": {"images": [{"image_url": {"url": uri}}]}}]}


class FakeResponse:
  def __init__(self, status_code=200, body=None, text=""):
    self.status_code = status_code
    self._body = body or {}
    self.text = text

  def json(self):
    return self._body


class FakeHttpx:
  """Stands in for httpx.Client; records every post."""

  def __init__(self, response: FakeResponse):
    self.response = response
    self.posts = []

  def factory(self, **kwargs):
    return self

  def __enter__(self):
    return self

  def __exit__(self, *_exc):
    return False

  def post(self, url, **kwargs):
    self.posts.append((url, kwargs))
    return self.response


@pytest.fixture
def api_keys(monkeypatch):
  monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")


@pytest.fixture
def fake_post(monkeypatch, api_keys):
  """Install a FakeHttpx serving `response` and return it."""

  def install(response: FakeResponse) -> FakeHttpx:
    fake = FakeHttpx(response)
    monkeypatch.setattr(image_module.httpx, "Client", fake.factory)
    return fake

  return install


# --- generate_image dispatch ---

class TestGenerateImageDispatch:
  def test_openai_prefix_hits_openai_generations(self, fake_post):
    fake = fake_post(FakeResponse(body=_openai_body()))

    result = generate_image("a red square", model="openai/gpt-image-2")

    assert result.mode == "RGB"
    url, kwargs = fake.posts[0]
    assert url == OPENAI_GENERATIONS
    assert kwargs["json"]["model"] == "gpt-image-2"  # prefix stripped
    assert kwargs["headers"]["Authorization"] == "Bearer sk-openai-test"

  def test_non_openai_model_hits_openrouter(self, fake_post):
    fake = fake_post(FakeResponse(body=_openrouter_body()))

    result = generate_image(
      "a red square", model="google/gemini-3.1-flash-image-preview",
    )

    assert result.mode == "RGB"
    url, kwargs = fake.posts[0]
    assert url == OPENROUTER_CHAT
    assert kwargs["json"]["model"] == "google/gemini-3.1-flash-image-preview"
    assert kwargs["json"]["modalities"] == ["image", "text"]
    assert kwargs["headers"]["Authorization"] == "Bearer sk-or-test"

  def test_openai_error_status_propagates(self, fake_post):
    fake_post(FakeResponse(status_code=500, text="server exploded"))

    with pytest.raises(RuntimeError, match="OpenAI API error 500"):
      generate_image("x", model="openai/gpt-image-2")

  def test_openrouter_error_status_propagates(self, fake_post):
    fake_post(FakeResponse(status_code=429, text="rate limited"))

    with pytest.raises(RuntimeError, match="OpenRouter API error 429"):
      generate_image("x", model="google/gemini-3.1-flash-image-preview")

  def test_openrouter_response_without_image_raises(self, fake_post):
    fake_post(FakeResponse(body={"choices": [{"message": {"content": "text only"}}]}))

    with pytest.raises(RuntimeError, match="No image in OpenRouter response"):
      generate_image("x", model="google/gemini-3.1-flash-image-preview")

  def test_missing_openrouter_key_raises(self, fake_post, monkeypatch):
    fake = fake_post(FakeResponse(body=_openrouter_body()))
    monkeypatch.delenv("OPENROUTER_API_KEY")

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
      generate_image("x", model="google/gemini-3.1-flash-image-preview")
    assert fake.posts == []

  def test_missing_openai_key_raises(self, fake_post, monkeypatch):
    fake = fake_post(FakeResponse(body=_openai_body()))
    monkeypatch.delenv("OPENAI_API_KEY")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
      generate_image("x", model="openai/gpt-image-2")
    assert fake.posts == []


# --- edit_image dispatch ---

class TestEditImageDispatch:
  @pytest.fixture
  def source(self):
    return Image.new("RGB", (10, 20), (200, 10, 10))

  def test_openai_prefix_hits_edits_endpoint_multipart(self, fake_post, source):
    fake = fake_post(FakeResponse(body=_openai_body()))

    result = edit_image(source, "make it blue", model="openai/gpt-image-2")

    url, kwargs = fake.posts[0]
    assert url == OPENAI_EDITS
    assert "image" in kwargs["files"]  # multipart upload, not JSON
    assert kwargs["data"]["model"] == "gpt-image-2"
    assert result.size == (10, 20)  # resized back to the input dimensions

  def test_non_openai_model_hits_openrouter_with_image_part(self, fake_post, source):
    fake = fake_post(FakeResponse(body=_openrouter_body()))

    result = edit_image(source, "make it blue", model="google/gemini-3.1-flash-image-preview")

    url, kwargs = fake.posts[0]
    assert url == OPENROUTER_CHAT
    content = kwargs["json"]["messages"][0]["content"]
    part_types = [part["type"] for part in content]
    assert part_types == ["text", "image_url"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert result.size == (10, 20)

  def test_openai_edit_error_status_propagates(self, fake_post, source):
    fake_post(FakeResponse(status_code=503, text="down"))

    with pytest.raises(RuntimeError, match="OpenAI edit API error 503"):
      edit_image(source, "x", model="openai/gpt-image-2")

  def test_openrouter_edit_error_status_propagates(self, fake_post, source):
    fake_post(FakeResponse(status_code=502, text="bad gateway"))

    with pytest.raises(RuntimeError, match="OpenRouter API error 502"):
      edit_image(source, "x", model="google/gemini-3.1-flash-image-preview")
