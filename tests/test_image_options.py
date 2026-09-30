"""Options that must reach the provider instead of being dropped on the way.

httpx is faked; the tests assert on the request payload.
"""

import base64
import io

import pytest
from PIL import Image

from merceka_core import image as image_module
from merceka_core.image import generate_image


def _png_b64(size=(8, 8)) -> str:
  buf = io.BytesIO()
  Image.new("RGB", size, (10, 20, 30)).save(buf, format="PNG")
  return base64.b64encode(buf.getvalue()).decode("ascii")


class FakeResponse:
  def __init__(self, body: dict):
    self.status_code = 200
    self._body = body
    self.text = ""

  def json(self) -> dict:
    return self._body


class FakeClient:
  def __init__(self, body: dict):
    self.body = body
    self.posts: list[tuple[str, dict]] = []

  def factory(self, *_args, **_kwargs):
    return self

  def __enter__(self):
    return self

  def __exit__(self, *_exc):
    return False

  def post(self, url, **kwargs):
    self.posts.append((url, kwargs))
    return FakeResponse(self.body)


@pytest.fixture
def fake_client(monkeypatch):
  monkeypatch.delenv("MERCEKA_FORCE_OPENROUTER", raising=False)

  def install(body: dict) -> FakeClient:
    fake = FakeClient(body)
    monkeypatch.setattr(image_module.httpx, "Client", fake.factory)
    return fake

  return install


def _gemini_body() -> dict:
  return {"candidates": [{"content": {"parts": [{"inlineData": {"data": _png_b64()}}]}}]}


@pytest.fixture
def google_direct(monkeypatch, fake_client):
  monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
  return fake_client(_gemini_body())


def _image_config(fake: FakeClient) -> dict:
  return fake.posts[0][1]["json"]["generationConfig"]["imageConfig"]


def test_google_direct_generation_sends_the_requested_image_size(google_direct):
  generate_image(
    "p", model="google/gemini-3-pro-image-preview", aspect_ratio="16:9", image_size="4K",
  )

  assert _image_config(google_direct) == {"aspectRatio": "16:9", "imageSize": "4K"}


def test_google_direct_generation_normalizes_the_image_size_tier(google_direct):
  generate_image("p", model="google/gemini-3.1-flash-image-preview", image_size="2k")

  assert _image_config(google_direct)["imageSize"] == "2K"


def test_google_direct_omits_image_size_for_a_model_without_it(google_direct):
  # Gemini 2.5 Flash Image documents no imageSize; only the aspect ratio goes.
  generate_image("p", model="google/gemini-2.5-flash-image", aspect_ratio="9:16", image_size="2K")

  assert _image_config(google_direct) == {"aspectRatio": "9:16"}


def test_google_direct_omits_tiers_flash_lite_cannot_serve(google_direct):
  # Gemini 3.1 Flash Lite Image serves 1K only.
  generate_image("p", model="google/gemini-3.1-flash-lite-image-preview", image_size="4K")

  assert "imageSize" not in _image_config(google_direct)
