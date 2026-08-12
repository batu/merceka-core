"""Cost ledger: recording, rates, summarize."""

import json

from merceka_core import costs


def test_record_and_summarize(tmp_path, monkeypatch):
  ledger = tmp_path / "costs.jsonl"
  monkeypatch.setenv("MERCEKA_COST_LEDGER", str(ledger))
  costs.record(source="openrouter", model="m1", usage={"total_tokens": 10}, usd=0.02)
  costs.record(source="google-direct", model="m2", usage={"candidatesTokenCount": 5})
  s = costs.summarize()
  assert s["calls"] == 2
  assert s["usd_known"] == 0.02
  assert s["usd_unknown_calls"] == 1
  rows = [json.loads(line) for line in ledger.read_text().splitlines()]
  assert rows[0]["usd"] == 0.02


def test_google_usage_priced_from_rates(tmp_path, monkeypatch):
  ledger = tmp_path / "costs.jsonl"
  monkeypatch.setenv("MERCEKA_COST_LEDGER", str(ledger))
  costs.record(
    source="google-direct",
    model="google/gemini-3.1-flash-image-preview",
    usage={"promptTokenCount": 1000, "candidatesTokenCount": 1120},
  )
  s = costs.summarize()
  # 1000/1M*0.50 + 1120/1M*60 = 0.0005 + 0.0672
  assert abs(s["usd_known"] - 0.0677) < 1e-6
  assert s["usd_unknown_calls"] == 0


def test_since_filter(tmp_path, monkeypatch):
  ledger = tmp_path / "costs.jsonl"
  monkeypatch.setenv("MERCEKA_COST_LEDGER", str(ledger))
  costs.record(source="openrouter", model="m", usage={}, usd=1.0)
  assert costs.summarize(since="2999-01-01")["calls"] == 0


def test_record_never_raises(monkeypatch):
  monkeypatch.setenv("MERCEKA_COST_LEDGER", "/dev/null/impossible/costs.jsonl")
  costs.record(source="x", model="y", usage={})  # must not raise


def test_attribution_context_tags_records(tmp_path, monkeypatch):
    """Cost attribution: records inside costs.attribution(...) carry the
    ambient meta, merged under any call-site meta; outside the block nothing
    is tagged."""
    import json

    from merceka_core import costs

    monkeypatch.setenv("MERCEKA_COST_LEDGER", str(tmp_path / "ledger.jsonl"))
    with costs.attribution({"sessionId": "level_a", "operation": "extract"}):
        costs.record(source="test", model="m", usage={}, usd=0.01,
                     meta={"birdId": "bird_1"})
    costs.record(source="test", model="m", usage={}, usd=0.02)
    rows = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert rows[0]["meta"] == {"sessionId": "level_a", "operation": "extract", "birdId": "bird_1"}
    assert "meta" not in rows[1]
