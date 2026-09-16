from merceka_core.image import _openai_size


def test_openai_1k_uses_fixed_aspect_ratio_size():
  assert _openai_size("16:9", "1K") == "1536x1024"
  assert _openai_size("9:16", "1K") == "1024x1536"
  assert _openai_size("1:1", "1K") == "1024x1024"


def test_gpt_image_2_uses_explicit_larger_sizes():
  assert _openai_size("16:9", "2K", "gpt-image-2") == "2560x1440"
  assert _openai_size("9:16", "2K", "gpt-image-2") == "1440x2560"
  assert _openai_size("16:9", "4K", "gpt-image-2") == "3840x2160"
  assert _openai_size("9:16", "4K", "gpt-image-2") == "2160x3840"
  assert _openai_size("1:1", "4K", "gpt-image-2") == "2880x2880"


def test_older_openai_larger_builder_tiers_use_auto():
  assert _openai_size("16:9", "2K", "gpt-image-1") == "auto"
  assert _openai_size("16:9", "4K", "gpt-image-1") == "auto"
  assert _openai_size("16:9", "auto") == "auto"


def test_openai_unknown_aspect_ratio_falls_back_to_square_1k():
  assert _openai_size("9:18", "1K") == "1024x1024"


def test_gpt_image_2_5_variants_share_the_explicit_size_table():
  assert _openai_size("1:1", "2K", "gpt-image-2.5-sunburst") == "2048x2048"
  assert _openai_size("1:1", "2K", "openai/gpt-image-2.5-flare") == "2048x2048"
  assert _openai_size("1:1", "2K", "gpt-image-1") == "auto"


def test_native_edit_size_only_for_gpt_image_2_family():
  from merceka_core.image import _openai_native_edit_size

  assert _openai_native_edit_size("openai/gpt-image-2.5-sunburst", 2048, 2048) == "2048x2048"
  assert _openai_native_edit_size("gpt-image-2", 1536, 2048) == "1536x2048"
  assert _openai_native_edit_size("gpt-image-1", 2048, 2048) is None
  assert _openai_native_edit_size("gpt-image-2", 2050, 2048) is None  # not a multiple of 16
  assert _openai_native_edit_size("gpt-image-2", 4096, 4096) is None  # over the 3840 edge cap
  assert _openai_native_edit_size("gpt-image-2", 3840, 1024) is None  # aspect beyond 3:1


def test_pad_to_multiple_of_16_replicates_edges_and_keeps_content_box():
  from PIL import Image

  from merceka_core.image import _pad_to_multiple_of_16

  im = Image.new("RGB", (182, 182), (10, 20, 30))
  im.putpixel((181, 0), (200, 0, 0))
  padded, box = _pad_to_multiple_of_16(im)
  assert padded.size == (192, 192)
  assert box == (0, 0, 182, 182)
  assert padded.getpixel((191, 0))[:3] == (200, 0, 0)  # right edge replicated
  assert padded.getpixel((0, 191))[:3] == (10, 20, 30)
  same, box2 = _pad_to_multiple_of_16(Image.new("RGB", (192, 96)))
  assert same.size == (192, 96) and box2 == (0, 0, 192, 96)


def test_native_edit_size_respects_the_minimum_pixel_budget():
  from merceka_core.image import _openai_native_edit_size

  assert _openai_native_edit_size("gpt-image-2.5-sunburst", 192, 192) is None  # API: below minimum pixel budget
  assert _openai_native_edit_size("gpt-image-2.5-sunburst", 768, 768) is None
  assert _openai_native_edit_size("gpt-image-2.5-sunburst", 896, 896) == "896x896"
