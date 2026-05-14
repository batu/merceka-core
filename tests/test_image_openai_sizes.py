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
