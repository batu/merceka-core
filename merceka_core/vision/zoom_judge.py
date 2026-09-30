"""Agentic zoom judge: an Anthropic-API panel member that can lean in.

Panel judges normally see each image once, downscaled to the patch budget —
small defects (thin halos, off-center glyphs, tiny text) can be illegible at
that size. This judge runs the zoom-tool loop from the Anthropic crop_tool
cookbook: images are pre-resized so the model's pixel coordinates map 1:1,
and a `zoom` tool crops requested regions from the full-resolution originals
and returns them magnified to fill the budget.

Talks to the Anthropic Messages API directly with httpx (no SDK dependency),
matching how the rest of the panel talks to OpenRouter. Key-gated on
ANTHROPIC_API_KEY by the caller in critique.py.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from merceka_core.vision.zoom import (
  MAX_EDGE_HIGHRES,
  MAX_TOKENS_HIGHRES,
  prepare_image,
  zoom_crop,
)

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_MAX_ROUNDS = 8
_MAX_OUTPUT_TOKENS = 4096
# Stop reasons that end a turn with a complete answer. max_tokens and refusal
# (and any reason added later) leave a truncated or absent verdict.
_FINAL_STOP_REASONS = frozenset({"end_turn", "stop_sequence"})
# Modes Pillow can write as PNG; anything else (CMYK, YCbCr, LAB, ...) is converted.
_PNG_MODES = frozenset({"1", "L", "LA", "I", "I;16", "P", "RGB", "RGBA"})

_ZOOM_TOOL = {
  "name": "zoom",
  "description": (
    "Zoom into a rectangular region of an image to see it magnified. Use this "
    "whenever text, edges, glyphs, or details are too small to judge "
    "confidently in the full view. The region is cropped from the image's "
    "full-resolution original and scaled up to fill the image budget. "
    "Coordinates are absolute pixels in the image as you see it, origin at "
    "the top-left corner. Zoom again on the result for even finer detail."
  ),
  "input_schema": {
    "type": "object",
    "properties": {
      "x1": {"type": "integer", "description": "Left edge in pixels"},
      "y1": {"type": "integer", "description": "Top edge in pixels"},
      "x2": {"type": "integer", "description": "Right edge in pixels (> x1)"},
      "y2": {"type": "integer", "description": "Bottom edge in pixels (> y1)"},
      "image_index": {
        "type": "integer",
        "description": "Which image to zoom into, by the numbered labels (starting at 0)",
      },
    },
    "required": ["x1", "y1", "x2", "y2"],
  },
}


def call_zoom_judge(
  judge: dict[str, Any],
  images: list[str | Path | bytes | bytearray | memoryview],
  reference: str | Path | None,
  prompt: str,
  *,
  api_key: str,
  client: httpx.Client | None = None,
  timeout: float = 300.0,
) -> dict[str, Any]:
  """Run the zoom-tool judging loop; returns raw response text or a skip reason.

  Returns {"ok": True, "text": <final text>} on success — the caller parses it
  with the panel's JSON verdict parser — or {"ok": False, "reason": ...}. Only a
  turn that ends with ``end_turn`` or ``stop_sequence`` succeeds; ``max_tokens``,
  ``refusal`` and other stop reasons return their name as the reason. Unreadable
  images and malformed API responses also return a reason instead of raising,
  so the panel keeps the judges that already finished.
  """
  try:
    originals, views, content = _first_turn(images, reference, prompt)
  except (OSError, ValueError, Image.DecompressionBombError):
    return {"ok": False, "reason": "image-error"}

  messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
  owns_client = client is None
  http_client = client or httpx.Client(timeout=timeout)
  try:
    for _ in range(_MAX_ROUNDS):
      try:
        response = http_client.post(
          ANTHROPIC_MESSAGES_URL,
          json={
            "model": judge["model"],
            "max_tokens": _MAX_OUTPUT_TOKENS,
            "tools": [_ZOOM_TOOL],
            "messages": messages,
          },
          headers={
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
          },
        )
      except httpx.TimeoutException:
        return {"ok": False, "reason": "timeout"}
      except httpx.HTTPError:
        return {"ok": False, "reason": "request-error"}
      if response.status_code != 200:
        return {"ok": False, "reason": f"anthropic-{response.status_code}"}

      try:
        body = response.json()
      except ValueError:
        return {"ok": False, "reason": "malformed-response"}
      blocks = _content_blocks(body)
      if blocks is None:
        return {"ok": False, "reason": "malformed-response"}
      messages.append({"role": "assistant", "content": blocks})
      stop_reason = body.get("stop_reason")

      if stop_reason == "tool_use":
        tool_uses = [block for block in blocks if block.get("type") == "tool_use"]
        if not tool_uses:
          return {"ok": False, "reason": "malformed-response"}
        results = [_tool_result(block, originals, views) for block in tool_uses]
        messages.append({"role": "user", "content": results})
        continue
      if stop_reason == "pause_turn":
        # Server paused a long turn; replaying the transcript resumes it.
        continue
      if stop_reason not in _FINAL_STOP_REASONS:
        # A truncated or refused turn has no trustworthy verdict.
        if not isinstance(stop_reason, str) or not stop_reason:
          return {"ok": False, "reason": "malformed-response"}
        return {"ok": False, "reason": stop_reason.replace("_", "-")}

      text = "\n".join(
        str(block.get("text", "")) for block in blocks if block.get("type") == "text"
      )
      if not text.strip():
        return {"ok": False, "reason": "empty-response"}
      return {"ok": True, "text": text}
    return {"ok": False, "reason": "max-rounds"}
  finally:
    if owns_client:
      http_client.close()


def _first_turn(
  images: list[str | Path | bytes | bytearray | memoryview],
  reference: str | Path | None,
  prompt: str,
) -> tuple[list[Image.Image], list[Image.Image], list[dict[str, Any]]]:
  originals: list[Image.Image] = []
  views: list[Image.Image] = []
  for image in images if reference is None else [reference, *images]:
    original = _load_image(image)
    originals.append(original)
    views.append(prepare_image(original, MAX_EDGE_HIGHRES, MAX_TOKENS_HIGHRES))

  content: list[dict[str, Any]] = [{"type": "text", "text": _prompt_with_zoom_note(prompt, views)}]
  for index, view in enumerate(views):
    if reference is not None and index == 0:
      label = f"Image {index}: REFERENCE"
    else:
      label = f"Image {index}: OURS"
    content.append({"type": "text", "text": f"{label} ({view.width}x{view.height} px)"})
    content.append(_image_block(view, "image/png"))
  return originals, views, content


def _content_blocks(body: Any) -> list[dict[str, Any]] | None:
  """The message's content blocks, or None when the body is not a well-formed message."""
  blocks = body.get("content") if isinstance(body, dict) else None
  if not isinstance(blocks, list):
    return None
  for block in blocks:
    if not isinstance(block, dict):
      return None
    if block.get("type") == "tool_use" and not isinstance(block.get("input"), dict):
      return None
  return blocks


def _prompt_with_zoom_note(prompt: str, views: list[Image.Image]) -> str:
  return (
    f"{prompt}\n\n"
    "A `zoom` tool is available. Before judging small detail — thin edges, "
    "halos, glyph centering, small text — zoom into it rather than guessing "
    "from the full view. Each image's dimensions are stated next to it; zoom "
    "coordinates are absolute pixels in that image, origin top-left."
  )


def _tool_result(
  block: dict[str, Any],
  originals: list[Image.Image],
  views: list[Image.Image],
) -> dict[str, Any]:
  tool_input = block["input"]
  result: dict[str, Any] = {"type": "tool_result", "tool_use_id": block.get("id")}
  try:
    index = int(tool_input.get("image_index", 0) or 0)
    if not 0 <= index < len(originals):
      raise ValueError(f"image_index {index} is out of range")
    x1, y1, x2, y2 = (int(tool_input[key]) for key in ("x1", "y1", "x2", "y2"))
    box = (x1, y1, x2, y2)
    crop = zoom_crop(
      originals[index],
      (views[index].width, views[index].height),
      box,
      MAX_EDGE_HIGHRES,
      MAX_TOKENS_HIGHRES,
    )
  except (KeyError, TypeError, ValueError, OverflowError) as exc:
    result["content"] = [{"type": "text", "text": f"Error: {exc}"}]
    result["is_error"] = True
    return result
  # JPEG keeps accumulated tool results well under the API request size limit.
  result["content"] = [
    {"type": "text", "text": f"Magnified view of image {index}, region {list(box)}:"},
    _image_block(crop.convert("RGB"), "image/jpeg"),
  ]
  return result


def _image_block(image: Image.Image, mime_type: str) -> dict[str, Any]:
  if mime_type == "image/png" and image.mode not in _PNG_MODES:
    # CMYK, YCbCr, LAB and friends cannot be written as PNG.
    image = image.convert("RGBA" if {"A", "a"} & set(image.getbands()) else "RGB")
  buffer = BytesIO()
  image.save(buffer, format="JPEG" if mime_type == "image/jpeg" else "PNG")
  return {
    "type": "image",
    "source": {
      "type": "base64",
      "media_type": mime_type,
      "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    },
  }


def _load_image(image: str | Path | bytes | bytearray | memoryview) -> Image.Image:
  if isinstance(image, (bytes, bytearray, memoryview)):
    return Image.open(BytesIO(bytes(image)))
  return Image.open(Path(image))


__all__ = ["ANTHROPIC_MESSAGES_URL", "call_zoom_judge"]
