import base64
import io

from PIL import Image

from merceka_core.image import _extract_openrouter_image_ref, _openrouter_image_or_raise


def _data_uri() -> str:
  img = Image.new("RGB", (1, 1), (255, 0, 0))
  buf = io.BytesIO()
  img.save(buf, format="PNG")
  encoded = base64.b64encode(buf.getvalue()).decode("ascii")
  return f"data:image/png;base64,{encoded}"


def test_extracts_openrouter_message_images_shape():
  uri = _data_uri()
  data = {
    "choices": [{
      "message": {
        "images": [{"image_url": {"url": uri}}],
      },
    }],
  }

  assert _extract_openrouter_image_ref(data) == uri


def test_extracts_openrouter_content_part_shape():
  uri = _data_uri()
  data = {
    "choices": [{
      "message": {
        "content": [
          {"type": "text", "text": "done"},
          {"type": "image_url", "image_url": {"url": uri}},
        ],
      },
    }],
  }

  assert _extract_openrouter_image_ref(data) == uri


def test_openrouter_image_or_raise_decodes_data_uri():
  uri = _data_uri()
  data = {
    "choices": [{
      "message": {
        "images": [{"image_url": {"url": uri}}],
      },
    }],
  }

  img = _openrouter_image_or_raise(data)

  assert img.mode == "RGB"
  assert img.size == (1, 1)
