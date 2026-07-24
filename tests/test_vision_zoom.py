import pytest
from PIL import Image

from merceka_core.vision import zoom


def test_count_image_tokens_counts_28px_patches():
  assert zoom.count_image_tokens(28, 28) == 1
  assert zoom.count_image_tokens(29, 28) == 2
  assert zoom.count_image_tokens(1568, 1568) == 56 * 56


def test_resized_size_returns_fitting_images_unchanged():
  assert zoom.resized_size(800, 600) == (800, 600)


def test_resized_size_downscales_oversized_images_within_limits():
  width, height = zoom.resized_size(4000, 3000)
  assert width <= zoom.MAX_EDGE_STANDARD and height <= zoom.MAX_EDGE_STANDARD
  assert zoom.count_image_tokens(width, height) <= zoom.MAX_TOKENS_STANDARD
  assert width / height == pytest.approx(4000 / 3000, abs=0.01)


def test_resized_size_is_maximal():
  width, height = zoom.resized_size(4000, 3000)
  bigger_h = max(round((width + 1) * 3000 / 4000), 1)
  assert not zoom._fits(width + 1, bigger_h, zoom.MAX_EDGE_STANDARD, zoom.MAX_TOKENS_STANDARD)


def test_resized_size_handles_portrait_symmetrically():
  landscape = zoom.resized_size(4000, 3000)
  portrait = zoom.resized_size(3000, 4000)
  assert portrait == (landscape[1], landscape[0])


def test_zoom_size_upscales_small_sizes_to_budget():
  width, height = zoom.zoom_size(100, 50)
  assert width > 100
  assert width / height == pytest.approx(2.0, abs=0.05)
  assert zoom.count_image_tokens(width, height) <= zoom.MAX_TOKENS_STANDARD
  assert math_ceil_edge(width) <= zoom.MAX_EDGE_STANDARD


def math_ceil_edge(edge: int) -> int:
  return -(-edge // 28) * 28


def test_prepare_image_makes_coordinates_map_one_to_one():
  image = Image.new("RGB", (4000, 3000), "white")
  view = zoom.prepare_image(image)
  assert (view.width, view.height) == zoom.resized_size(4000, 3000)
  assert zoom.prepare_image(view) is view


def test_zoom_crop_magnifies_from_the_full_resolution_original():
  original = Image.new("RGB", (4000, 3000), "white")
  original.paste(Image.new("RGB", (400, 300), "red"), (2000, 1500))
  view_size = zoom.resized_size(4000, 3000)
  view_w, view_h = view_size
  crop = zoom.zoom_crop(
    original,
    view_size,
    (view_w // 2, view_h // 2, view_w // 2 + view_w // 10, view_h // 2 + view_h // 10),
  )
  assert crop.width > view_w // 10  # magnified, not returned at crop size
  assert crop.getpixel((1, 1)) == (255, 0, 0)


def test_zoom_crop_rejects_degenerate_boxes():
  original = Image.new("RGB", (100, 100))
  with pytest.raises(ValueError):
    zoom.zoom_crop(original, (100, 100), (50, 50, 50, 60))
  with pytest.raises(ValueError):
    zoom.zoom_crop(original, (100, 100), (120, 0, 140, 10))


def test_fill_budget_no_ops_at_target_size():
  target = zoom.zoom_size(200, 100)
  image = Image.new("RGB", target)
  assert zoom.fill_budget(image) is image
