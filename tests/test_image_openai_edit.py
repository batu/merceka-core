"""OpenAI-direct edit path, exercised end to end through ``edit_image``.

httpx.Client is replaced with a fake that answers each edit with an image of
the size the request asked for, the way the API does. The tests assert on the
request that would have been paid for and on the ledger rows it left behind.
"""

import base64
import io
import json

import pytest
from PIL import Image

from merceka_core import costs
from merceka_core import image as image_module
from merceka_core.image import edit_image

OPENAI_EDITS = "https://api.openai.com/v1/images/edits"
BLUE = (0, 0, 255)
RED = (255, 0, 0)


class FakeResponse:
  def __init__(self, body: dict):
    self.status_code = 200
    self._body = body
    self.text = json.dumps(body)[:200]

  def json(self) -> dict:
    return self._body


class FakeOpenAIEdits:
  """Answers an edit with a PNG of the requested ``size``.

  Columns at or beyond ``content_width`` are painted red, the rest blue, so a
  test can tell a crop of the padding (all blue) from a resize (red bleeds in).
  """

  def __init__(self, content_width: int | None = None):
    self.content_width = content_width
    self.posts: list[tuple[str, dict]] = []

  def factory(self, *_args, **_kwargs):
    return self

  def __enter__(self):
    return self

  def __exit__(self, *_exc):
    return False

  def post(self, url, **kwargs):
    self.posts.append((url, kwargs))
    w, h = (int(v) for v in kwargs["data"]["size"].split("x"))
    img = Image.new("RGB", (w, h), BLUE)
    if self.content_width is not None and self.content_width < w:
      img.paste(RED, (self.content_width, 0, w, h))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return FakeResponse({
      "data": [{"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")}],
      "usage": {"input_tokens": 10, "output_tokens": 100},
    })

  def uploaded(self, index: int = 0, field: str = "image") -> Image.Image:
    return Image.open(io.BytesIO(self.posts[index][1]["files"][field][1]))


@pytest.fixture
def openai_edits(monkeypatch):
  monkeypatch.setenv("OPENAI_API_KEY", "test-key")

  def install(content_width: int | None = None) -> FakeOpenAIEdits:
    fake = FakeOpenAIEdits(content_width)
    monkeypatch.setattr(image_module.httpx, "Client", fake.factory)
    return fake

  return install


def _ledger_rows() -> list[dict]:
  path = costs.ledger_path()
  if not path.exists():
    return []
  return [json.loads(line) for line in path.read_text().splitlines()]


def test_edit_pads_to_a_multiple_of_16_and_requests_the_padded_native_size(openai_edits):
  # 1080 is not a multiple of 16. Padded to 1088x1920 the input is a native
  # gpt-image-2 size, so the request must carry it instead of a fixed 1K size.
  fake = openai_edits(content_width=1080)
  source = Image.new("RGB", (1080, 1920), BLUE)

  result = edit_image(source, "recolor", model="openai/gpt-image-2.5-sunburst")

  url, kwargs = fake.posts[0]
  assert url == OPENAI_EDITS
  assert kwargs["data"]["size"] == "1088x1920"
  assert fake.uploaded().size == (1088, 1920)
  assert result.size == (1080, 1920)
  # The padding is cropped away, not resized into the content.
  assert result.getpixel((1079, 0)) == BLUE
  assert len(_ledger_rows()) == 1


def test_edit_pads_both_edges_when_neither_is_a_multiple_of_16(openai_edits):
  fake = openai_edits(content_width=1200)

  result = edit_image(Image.new("RGB", (1200, 1000), BLUE), "p", model="openai/gpt-image-2")

  assert fake.posts[0][1]["data"]["size"] == "1200x1008"
  assert result.size == (1200, 1000)


def test_native_size_input_is_sent_unpadded(openai_edits):
  fake = openai_edits()

  result = edit_image(Image.new("RGB", (1024, 768), BLUE), "p", model="openai/gpt-image-2")

  assert fake.posts[0][1]["data"]["size"] == "1024x768"
  assert fake.uploaded().size == (1024, 768)
  assert result.size == (1024, 768)


def test_small_square_sticker_uses_the_fixed_square_size(openai_edits):
  # 182 px pads to 192x192, still under the minimum pixel budget: the fixed
  # 1024x1024 size serves it at the same aspect.
  fake = openai_edits()

  result = edit_image(Image.new("RGB", (182, 182), BLUE), "p", model="openai/gpt-image-2.5-sunburst")

  assert fake.posts[0][1]["data"]["size"] == "1024x1024"
  assert fake.uploaded().size == (182, 182)
  assert result.size == (182, 182)


def test_unservable_aspect_fails_before_the_paid_call(openai_edits):
  # 120x100 pads to 128x112, under the pixel floor; the nearest fixed size is
  # 1536x1024 (3:2), which the aspect guard would refuse after charging.
  fake = openai_edits()

  with pytest.raises(ValueError, match="before the paid call"):
    edit_image(Image.new("RGB", (120, 100), BLUE), "p", model="openai/gpt-image-2.5-sunburst")

  assert fake.posts == []
  assert _ledger_rows() == []
