import base64
import io
from unittest.mock import patch

from PIL import Image

from merceka_core.image import _generate_openai, _openrouter_image_or_raise, generate_image


def _png_uri(mode: str, color) -> str:
  img = Image.new(mode, (2, 2), color)
  buf = io.BytesIO()
  img.save(buf, format="PNG")
  encoded = base64.b64encode(buf.getvalue()).decode("ascii")
  return f"data:image/png;base64,{encoded}"


def _openrouter_response(uri: str) -> dict:
  return {"choices": [{"message": {"images": [{"image_url": {"url": uri}}]}}]}


class _FakeResponse:
  status_code = 200

  def __init__(self, payload: dict):
    self._payload = payload

  def json(self) -> dict:
    return self._payload


class _CapturingClient:
  """httpx.Client stand-in capturing the request payload."""

  last_payload: dict | None = None
  response_payload: dict = {}

  def __init__(self, *_args, **_kwargs):
    pass

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    return False

  def post(self, url, json=None, **_kwargs):
    type(self).last_payload = json
    return _FakeResponse(type(self).response_payload)


def test_openrouter_decode_preserves_alpha_when_transparent():
  uri = _png_uri("RGBA", (255, 0, 0, 128))
  img = _openrouter_image_or_raise(_openrouter_response(uri), transparent=True)
  assert img.mode == "RGBA"
  assert img.getpixel((0, 0))[3] == 128


def test_openrouter_decode_default_stays_rgb():
  uri = _png_uri("RGBA", (255, 0, 0, 128))
  img = _openrouter_image_or_raise(_openrouter_response(uri))
  assert img.mode == "RGB"


def test_openrouter_transparent_without_alpha_returns_rgb():
  uri = _png_uri("RGB", (255, 0, 0))
  img = _openrouter_image_or_raise(_openrouter_response(uri), transparent=True)
  assert img.mode == "RGB"


def test_generate_image_openrouter_transparent_prompt_and_alpha(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  _CapturingClient.response_payload = _openrouter_response(_png_uri("RGBA", (0, 255, 0, 64)))
  with patch("merceka_core.image.httpx.Client", _CapturingClient):
    img = generate_image("a coin", model="google/gemini-3.1-flash-image", transparent=True)
  content = _CapturingClient.last_payload["messages"][0]["content"]
  assert "transparent background" in content
  assert img.mode == "RGBA"


def test_generate_image_openrouter_default_has_no_transparent_suffix(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  _CapturingClient.response_payload = _openrouter_response(_png_uri("RGB", (0, 255, 0)))
  with patch("merceka_core.image.httpx.Client", _CapturingClient):
    img = generate_image("a coin", model="google/gemini-3.1-flash-image")
  content = _CapturingClient.last_payload["messages"][0]["content"]
  assert "transparent background" not in content
  assert img.mode == "RGB"


def _openai_b64(mode: str, color) -> dict:
  img = Image.new(mode, (2, 2), color)
  buf = io.BytesIO()
  img.save(buf, format="PNG")
  return {"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")}]}


def test_generate_openai_transparent_payload_and_rgba(monkeypatch):
  monkeypatch.setenv("OPENAI_API_KEY", "test-key")
  _CapturingClient.response_payload = _openai_b64("RGBA", (0, 0, 255, 30))
  with patch("merceka_core.image.httpx.Client", _CapturingClient):
    img = _generate_openai("a coin", "gpt-image-2", "1:1", "1K", transparent=True)
  assert _CapturingClient.last_payload["background"] == "transparent"
  assert img.mode == "RGBA"
  assert img.getpixel((0, 0))[3] == 30


def test_generate_openai_default_payload_unchanged(monkeypatch):
  monkeypatch.setenv("OPENAI_API_KEY", "test-key")
  _CapturingClient.response_payload = _openai_b64("RGB", (0, 0, 255))
  with patch("merceka_core.image.httpx.Client", _CapturingClient):
    img = _generate_openai("a coin", "gpt-image-2", "1:1", "1K")
  assert "background" not in _CapturingClient.last_payload
  assert img.mode == "RGB"


def test_openai_prefix_falls_back_to_openrouter_without_key(monkeypatch):
  monkeypatch.delenv("OPENAI_API_KEY", raising=False)
  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  _CapturingClient.response_payload = _openrouter_response(_png_uri("RGB", (9, 9, 9)))
  with patch("merceka_core.image.httpx.Client", _CapturingClient):
    img = generate_image("a coin", model="openai/gpt-5.4-image-2")
  assert _CapturingClient.last_payload["model"] == "openai/gpt-5.4-image-2"
  assert _CapturingClient.last_payload["modalities"] == ["image", "text"]
  assert img.mode == "RGB"


def test_edit_image_can_skip_resize_and_falls_back_without_key(monkeypatch):
  from merceka_core.image import edit_image

  monkeypatch.delenv("OPENAI_API_KEY", raising=False)
  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  _CapturingClient.response_payload = _openrouter_response(_png_uri("RGB", (9, 9, 9)))
  anchor = Image.new("RGB", (64, 64), (1, 2, 3))
  with patch("merceka_core.image.httpx.Client", _CapturingClient):
    kept = edit_image(anchor, "make tiers", model="openai/gpt-5.4-image-2")
    free = edit_image(anchor, "make tiers", model="openai/gpt-5.4-image-2", resize_to_input=False)
  assert _CapturingClient.last_payload["model"] == "openai/gpt-5.4-image-2"
  assert kept.size == (64, 64)
  assert free.size == (2, 2)  # model output size preserved
