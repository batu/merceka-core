"""Patch-budget image sizing and magnified crops for vision models.

Vision models see images in 28x28-pixel patches under a per-image budget;
oversized images are downscaled server-side before the model sees them, and
small details scale down with them. These helpers make that budget explicit:

- ``prepare_image`` pre-resizes an image to exactly the size the model will
  see, so pixel coordinates the model emits map 1:1 onto the image we hold.
- ``zoom_crop`` cuts a region out of the full-resolution original and scales
  it up to fill the budget, so small details become legible ("lean in").

Reference: the resized_size() implementation from the Claude vision
coordinates guide and the zoom-tool cookbook
(anthropics/claude-cookbooks multimodal/crop_tool.ipynb). Defaults use the
standard resolution tier, which is safe for every model family the studio's
judges run on; pass the high-resolution limits for models that support them.
"""

from __future__ import annotations

import math

from PIL import Image

# Standard resolution tier (safe across model families).
MAX_EDGE_STANDARD = 1568
MAX_TOKENS_STANDARD = 1568
# High-resolution tier (e.g. claude-fable-5, claude-sonnet-5).
MAX_EDGE_HIGHRES = 2576
MAX_TOKENS_HIGHRES = 4784


def count_image_tokens(width: int, height: int) -> int:
  """Visual tokens consumed by an image: one token per 28x28 pixel patch."""
  return math.ceil(width / 28) * math.ceil(height / 28)


def _fits(width: int, height: int, max_edge: int, max_tokens: int) -> bool:
  return (
    math.ceil(width / 28) * 28 <= max_edge
    and math.ceil(height / 28) * 28 <= max_edge
    and count_image_tokens(width, height) <= max_tokens
  )


def resized_size(
  width: int,
  height: int,
  max_edge: int = MAX_EDGE_STANDARD,
  max_tokens: int = MAX_TOKENS_STANDARD,
) -> tuple[int, int]:
  """The size the model resizes an image to before patching.

  Images that already fit within the limits are returned unchanged.
  """
  if _fits(width, height, max_edge, max_tokens):
    return (width, height)
  if height > width:
    resized_h, resized_w = resized_size(height, width, max_edge, max_tokens)
    return (resized_w, resized_h)

  aspect_ratio = width / height
  lo, hi = 1, width  # lo always fits; hi never fits
  while lo + 1 < hi:
    mid = (lo + hi) // 2
    if _fits(mid, max(round(mid / aspect_ratio), 1), max_edge, max_tokens):
      lo = mid
    else:
      hi = mid
  return (lo, max(round(lo / aspect_ratio), 1))


def prepare_image(
  image: Image.Image,
  max_edge: int = MAX_EDGE_STANDARD,
  max_tokens: int = MAX_TOKENS_STANDARD,
) -> Image.Image:
  """Resize an image to exactly the size the model will see.

  Pixel coordinates a model emits refer to the image it saw — after any
  server-side downscaling. Doing the same resize client-side first means the
  model's coordinates map 1:1 onto the image we hold.
  """
  target = resized_size(image.width, image.height, max_edge, max_tokens)
  if (image.width, image.height) == target:
    return image
  return image.resize(target, Image.Resampling.LANCZOS)


def zoom_size(
  width: int,
  height: int,
  max_edge: int = MAX_EDGE_STANDARD,
  max_tokens: int = MAX_TOKENS_STANDARD,
) -> tuple[int, int]:
  """The largest aspect-preserving size within the image limits.

  The upscale target for a magnified crop: the larger the crop is drawn, the
  more 28x28 patches each element inside it covers.
  """
  if height > width:
    zoom_h, zoom_w = zoom_size(height, width, max_edge, max_tokens)
    return (zoom_w, zoom_h)

  aspect_ratio = width / height
  lo, hi = 1, max_edge + 1  # lo always fits; hi never fits
  while lo + 1 < hi:
    mid = (lo + hi) // 2
    if _fits(mid, max(round(mid / aspect_ratio), 1), max_edge, max_tokens):
      lo = mid
    else:
      hi = mid
  return (lo, max(round(lo / aspect_ratio), 1))


def fill_budget(
  image: Image.Image,
  max_edge: int = MAX_EDGE_STANDARD,
  max_tokens: int = MAX_TOKENS_STANDARD,
) -> Image.Image:
  """Scale an image up (or down) to the largest size within the image limits."""
  target = zoom_size(image.width, image.height, max_edge, max_tokens)
  if (image.width, image.height) == target:
    return image
  return image.resize(target, Image.Resampling.LANCZOS)


def zoom_crop(
  original: Image.Image,
  view_size: tuple[int, int],
  box: tuple[int, int, int, int],
  max_edge: int = MAX_EDGE_STANDARD,
  max_tokens: int = MAX_TOKENS_STANDARD,
) -> Image.Image:
  """Crop ``box`` (view-space pixels) from the full-resolution original, magnified.

  ``box`` is (x1, y1, x2, y2) in the coordinate space of the view the model
  saw (``view_size``, typically from :func:`prepare_image`); the region is
  mapped onto the original — which may be larger — cropped from it so every
  source pixel contributes, then scaled to fill the image budget.
  """
  x1, y1, x2, y2 = box
  view_w, view_h = view_size
  if view_w <= 0 or view_h <= 0:
    raise ValueError("view_size must be positive")
  scale_x = original.width / view_w
  scale_y = original.height / view_h
  left = max(0, min(original.width, round(x1 * scale_x)))
  top = max(0, min(original.height, round(y1 * scale_y)))
  right = max(0, min(original.width, round(x2 * scale_x)))
  bottom = max(0, min(original.height, round(y2 * scale_y)))
  if right <= left or bottom <= top:
    raise ValueError(f"zoom box {box} does not describe a region inside the view {view_size}")
  crop = original.crop((left, top, right, bottom))
  return fill_budget(crop, max_edge, max_tokens)
