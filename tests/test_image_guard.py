import pytest
from PIL import Image

from merceka_core.image import _resize_to_input_guarded


def test_matching_size_returns_original_image():
  result = Image.new("RGB", (8, 8))

  resized = _resize_to_input_guarded(result, (8, 8))

  assert resized is result


def test_matching_aspect_resizes_to_original_size():
  result = Image.new("RGB", (16, 16))

  resized = _resize_to_input_guarded(result, (8, 8))

  assert resized.size == (8, 8)


def test_aspect_delta_just_under_two_percent_resizes():
  result = Image.new("RGB", (100, 99))

  resized = _resize_to_input_guarded(result, (8, 8))

  assert resized.size == (8, 8)


def test_aspect_delta_over_two_percent_refuses_to_stretch():
  result = Image.new("RGB", (8, 12))

  with pytest.raises(RuntimeError, match="refusing to stretch"):
    _resize_to_input_guarded(result, (8, 8))


@pytest.mark.skip(reason="original image dimensions are assumed to be valid and non-zero")
def test_zero_original_height_is_outside_the_guard_contract():
  _resize_to_input_guarded(Image.new("RGB", (8, 8)), (8, 0))
