"""Cost ledger: recording, rates, provenance, summarize."""

import json

import pytest

from merceka_core import costs


def _rows(path):
  return [json.loads(line) for line in path.read_text().splitlines()]


def test_record_and_summarize(tmp_path, monkeypatch):
  ledger = tmp_path / "costs.jsonl"
  monkeypatch.setenv("MERCEKA_COST_LEDGER", str(ledger))
  costs.record(source="openrouter", model="m1", usage={"total_tokens": 10}, usd=0.02)
  costs.record(source="google-direct", model="m2", usage={"candidatesTokenCount": 5})
  s = costs.summarize()
  assert s["calls"] == 2
  assert s["usd_known"] == 0.02
  assert s["usd_metered"] == 0.02
  assert s["usd_rates"] == 0.0
  assert s["usd_unknown_calls"] == 1
  rows = _rows(ledger)
  assert rows[0]["usd"] == 0.02
  assert rows[0]["usd_source"] == "provider"
  assert "usd_source" not in rows[1]


# Real google-direct usageMetadata shapes from ~/.merceka/costs.jsonl.
GEMINI_IMAGE_USAGE = {
  "promptTokenCount": 402,
  "candidatesTokenCount": 1415,
  "candidatesTokensDetails": [{"modality": "IMAGE", "tokenCount": 1120}],
  "totalTokenCount": 1817,
}
GEMINI_THINKING_USAGE = {
  "promptTokenCount": 402,
  "candidatesTokenCount": 1248,
  "thoughtsTokenCount": 153,
  "candidatesTokensDetails": [{"modality": "IMAGE", "tokenCount": 1120}],
  "totalTokenCount": 1803,
}


def test_google_usage_priced_by_output_modality(tmp_path, monkeypatch):
  """Image output is $60/M, text and thinking output $3/M (Gemini pricing page).

  Pricing all of candidatesTokenCount at $60/M overstated google-direct spend ~19%.
  """
  ledger = tmp_path / "costs.jsonl"
  monkeypatch.setenv("MERCEKA_COST_LEDGER", str(ledger))
  costs.record(
    source="google-direct", model="google/gemini-3.1-flash-image-preview",
    usage=GEMINI_IMAGE_USAGE,
  )
  expected = round(402 / 1e6 * 0.5 + 1120 / 1e6 * 60 + (1415 - 1120) / 1e6 * 3, 6)
  row = _rows(ledger)[0]
  assert row["usd"] == expected
  assert row["usd_source"] == "rates"
  s = costs.summarize()
  assert s["usd_rates"] == expected
  assert s["usd_metered"] == 0.0


def test_google_thinking_tokens_are_billed_as_text_output():
  usd = costs._usd_from_rates("google/gemini-3.1-flash-image-preview", GEMINI_THINKING_USAGE)
  text_out = 1248 - 1120 + 153
  assert usd == round(402 / 1e6 * 0.5 + 1120 / 1e6 * 60 + text_out / 1e6 * 3, 6)


def test_google_usage_without_modality_breakdown_stays_unpriced():
  usage = {"promptTokenCount": 402, "candidatesTokenCount": 1415}
  assert costs._usd_from_rates("google/gemini-3.1-flash-image-preview", usage) is None


def test_partial_usage_is_not_priced():
  """A row missing a rated field used to be priced from the fields present."""
  usage = {"input_tokens_details": {"text_tokens": 100}, "output_tokens": 50}
  assert costs._usd_from_rates("openai/gpt-image-2", usage) is None


def test_non_json_meta_does_not_drop_the_row(tmp_path, monkeypatch):
  ledger = tmp_path / "costs.jsonl"
  monkeypatch.setenv("MERCEKA_COST_LEDGER", str(ledger))
  with costs.attribution({"level": tmp_path / "level.json"}):
    costs.record(source="openrouter", model="m", usage={}, usd=0.5, meta={"n": {1, 2}})
  row = _rows(ledger)[0]
  assert row["usd"] == 0.5
  assert row["meta"]["level"] == str(tmp_path / "level.json")


def test_request_id_is_recorded(tmp_path, monkeypatch):
  ledger = tmp_path / "costs.jsonl"
  monkeypatch.setenv("MERCEKA_COST_LEDGER", str(ledger))
  costs.record(source="openrouter", model="m", usage={}, usd=0.1, request_id="gen-123")
  assert _rows(ledger)[0]["request_id"] == "gen-123"


def test_since_filter(tmp_path, monkeypatch):
  ledger = tmp_path / "costs.jsonl"
  monkeypatch.setenv("MERCEKA_COST_LEDGER", str(ledger))
  costs.record(source="openrouter", model="m", usage={}, usd=1.0)
  assert costs.summarize(since="2999-01-01")["calls"] == 0


def test_since_compares_instants_not_strings(tmp_path, monkeypatch):
  """08:30Z is after 10:00+03:00 (07:00Z); a string compare dropped it."""
  ledger = tmp_path / "costs.jsonl"
  ledger.write_text(
    json.dumps({"ts": "2026-09-30T08:30:00+00:00", "source": "x", "model": "m",
                "usage": {}, "usd": 1.0}) + "\n"
    + json.dumps({"ts": "2026-09-30T06:30:00+00:00", "source": "x", "model": "m",
                  "usage": {}, "usd": 2.0}) + "\n"
  )
  monkeypatch.setenv("MERCEKA_COST_LEDGER", str(ledger))
  assert costs.summarize(since="2026-09-30T10:00:00+03:00")["usd_known"] == 1.0
  assert costs.summarize(since="2026-09-30T07:00:00Z")["usd_known"] == 1.0
  assert costs.summarize(since="2026-09-30")["calls"] == 2


def test_since_rejects_garbage(tmp_path, monkeypatch):
  monkeypatch.setenv("MERCEKA_COST_LEDGER", str(tmp_path / "costs.jsonl"))
  with pytest.raises(ValueError):
    costs.summarize(since="yesterday")


def test_legacy_rows_are_attributed_by_source(tmp_path, monkeypatch):
  """Rows written before usd_source existed: openai-/google-direct were rates."""
  ledger = tmp_path / "costs.jsonl"
  ledger.write_text(
    json.dumps({"ts": "2026-09-01T00:00:00+00:00", "source": "openai-direct",
                "model": "openai/gpt-image-2", "usage": {}, "usd": 0.25}) + "\n"
    + json.dumps({"ts": "2026-09-01T00:00:00+00:00", "source": "openrouter",
                  "model": "m", "usage": {}, "usd": 0.75}) + "\n"
  )
  monkeypatch.setenv("MERCEKA_COST_LEDGER", str(ledger))
  s = costs.summarize()
  assert (s["usd_rates"], s["usd_metered"], s["usd_known"]) == (0.25, 0.75, 1.0)


def test_record_never_raises(monkeypatch):
  monkeypatch.setenv("MERCEKA_COST_LEDGER", "/dev/null/impossible/costs.jsonl")
  costs.record(source="x", model="y", usage={})  # must not raise


def test_attribution_context_tags_records(tmp_path, monkeypatch):
  """Records inside costs.attribution(...) carry the ambient meta, merged under
  any call-site meta; outside the block nothing is tagged."""
  monkeypatch.setenv("MERCEKA_COST_LEDGER", str(tmp_path / "ledger.jsonl"))
  with costs.attribution({"sessionId": "level_a", "operation": "extract"}):
    costs.record(source="test", model="m", usage={}, usd=0.01, meta={"birdId": "bird_1"})
  costs.record(source="test", model="m", usage={}, usd=0.02)
  rows = _rows(tmp_path / "ledger.jsonl")
  assert rows[0]["meta"] == {"sessionId": "level_a", "operation": "extract", "birdId": "bird_1"}
  assert "meta" not in rows[1]
