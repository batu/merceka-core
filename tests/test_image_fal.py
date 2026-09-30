"""fal.ai inpaint path (``inpaint`` with a ``fal-ai/`` model), with httpx faked."""

import io
import json

import pytest
from PIL import Image

from merceka_core import costs
from merceka_core import image as image_module
from merceka_core.image import inpaint

FAL_FILL = "fal-ai/flux-pro/v1/fill"
RESULT_URL = "https://fal.media/files/result.png"


def _png(size: tuple[int, int]) -> bytes:
  buf = io.BytesIO()
  Image.new("RGB", size, (0, 128, 0)).save(buf, format="PNG")
  return buf.getvalue()


class FakeResponse:
  def __init__(self, body=None, content=b"", headers=None, status_code=200):
    self.status_code = status_code
    self._body = body or {}
    self.content = content
    self.headers = headers or {}
    self.text = json.dumps(self._body)[:200]

  def json(self):
    return self._body

  def raise_for_status(self):
    if self.status_code != 200:
      raise RuntimeError(f"HTTP {self.status_code}")


class FakeFal:
  def __init__(self, body: dict, result_size=(300, 200), headers=None):
    self.body = body
    self.result_size = result_size
    self.headers = headers or {}
    self.posts: list[tuple[str, dict]] = []

  def factory(self, *_args, **_kwargs):
    return self

  def __enter__(self):
    return self

  def __exit__(self, *_exc):
    return False

  def post(self, url, **kwargs):
    self.posts.append((url, kwargs))
    return FakeResponse(body=self.body, headers=self.headers)

  def get(self, url, **_kwargs):
    assert url == RESULT_URL
    return FakeResponse(content=_png(self.result_size))


@pytest.fixture
def fal(monkeypatch):
  monkeypatch.setenv("FAL_KEY", "test-key")

  def install(body=None, result_size=(300, 200), headers=None) -> FakeFal:
    fake = FakeFal(
      body if body is not None else {"images": [{"url": RESULT_URL}]},
      result_size=result_size,
      headers=headers,
    )
    monkeypatch.setattr(image_module.httpx, "Client", fake.factory)
    return fake

  return install


def _ledger_rows() -> list[dict]:
  path = costs.ledger_path()
  if not path.exists():
    return []
  return [json.loads(line) for line in path.read_text().splitlines()]


def _scene(size=(300, 200)) -> tuple[Image.Image, Image.Image]:
  return Image.new("RGB", size, (10, 10, 10)), Image.new("L", size, 255)


def test_fal_inpaint_records_a_ledger_row(fal):
  fal(headers={"x-fal-request-id": "req-123"})
  image, mask = _scene()

  result = inpaint(image, mask, "fill the sky")

  assert result.size == (300, 200)
  rows = _ledger_rows()
  assert len(rows) == 1
  assert rows[0]["source"] == "fal"
  assert rows[0]["model"] == FAL_FILL
  assert rows[0]["usage"] == {"calls": 1}
  assert rows[0]["usd"] is None
  assert rows[0]["request_id"] == "req-123"


def test_fal_inpaint_records_the_call_even_when_no_image_comes_back(fal):
  fal(body={"images": []})
  image, mask = _scene()

  with pytest.raises(RuntimeError, match="No image in fal.ai response"):
    inpaint(image, mask, "fill the sky")

  assert [row["model"] for row in _ledger_rows()] == [FAL_FILL]
