from merceka_core.costs import _usd_from_rates


def test_openai_image_usage_is_priced_from_nested_token_details():
  usage = {
    "input_tokens": 378,
    "input_tokens_details": {"image_tokens": 24, "text_tokens": 354},
    "output_tokens": 158,
    "output_tokens_details": {"image_tokens": 158, "text_tokens": 0},
  }
  usd = _usd_from_rates("openai/gpt-image-2.5-sunburst", usage)
  assert usd == round(354 / 1e6 * 5 + 24 / 1e6 * 8 + 158 / 1e6 * 30, 6)
  assert _usd_from_rates("openai/gpt-image-2", usage) == usd


def test_unknown_model_or_missing_fields_stay_unpriced():
  assert _usd_from_rates("openai/nope", {"output_tokens": 5}) is None
  assert _usd_from_rates("openai/gpt-image-2", {}) is None
