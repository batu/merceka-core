"""Per-call provider cost ledger.

Every provider call that reports usage appends one JSONL row here. Token
counts come from the provider's own response (hard data, what billing is
computed from). ``usd`` is never guessed. It is filled in one of two ways:

- the provider states the cost (OpenRouter ``usage.cost``, grok's
  ``total_cost_usd``): ``usd_source: "provider"``;
- every priced field of a rate-table entry is present in the usage:
  ``usd_source: "rates"``. This is price-sheet arithmetic, an estimate that
  reports label as such.

Anything else is ``usd: null``, with the tokens kept so cost can be derived later.

Ledger path: $MERCEKA_COST_LEDGER, default ~/.merceka/costs.jsonl.
Summarize: `python -m merceka_core.costs [--since ISO8601]`.
"""

__all__ = ["attribution", "record", "summarize", "ledger_path"]

import contextlib
import contextvars
import datetime as _dt
import json
import os
from pathlib import Path

# USD per 1M tokens by (model, usage field). Only entries verified against the
# provider's current price sheet belong here; unknown models simply get
# usd=None rows (tokens are still recorded, so cost is derivable later).
RATES_PATH_ENV = "MERCEKA_RATES_PATH"
# Rows written before usd_source existed: these sources were always priced from
# rates.json, every other source passed the provider's own figure.
_LEGACY_RATE_SOURCES = frozenset({"openai-direct", "google-direct"})

_attribution_var: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
  "merceka_cost_attribution", default=None,
)


def ledger_path() -> Path:
  return Path(os.environ.get("MERCEKA_COST_LEDGER", "~/.merceka/costs.jsonl")).expanduser()


def _load_rates() -> dict:
  path = os.environ.get(RATES_PATH_ENV)
  candidates = [Path(path)] if path else [Path(__file__).parent / "rates.json"]
  for p in candidates:
    if p.exists():
      try:
        return json.loads(p.read_text())
      except (OSError, json.JSONDecodeError):
        return {}
  return {}


def _gemini_output_split(usage: dict) -> dict:
  """Derived output fields for Gemini ``usageMetadata``.

  Google bills image output tokens and text-and-thinking output tokens at
  different rates. ``candidatesTokensDetails`` lists the IMAGE tokens. The rest
  of ``candidatesTokenCount``, plus ``thoughtsTokenCount``, is text and thinking:
  ``totalTokenCount`` is prompt + candidates + thoughts. Without the modality
  breakdown the split is unknown, so no fields are derived and the row stays
  unpriced.
  """
  details = usage.get("candidatesTokensDetails")
  candidates = usage.get("candidatesTokenCount")
  if not isinstance(details, list) or not isinstance(candidates, (int, float)):
    return {}
  image = sum(
    d.get("tokenCount") or 0
    for d in details
    if isinstance(d, dict) and d.get("modality") == "IMAGE"
  )
  thoughts = usage.get("thoughtsTokenCount") or 0
  return {"outputImageTokens": image, "outputTextTokens": candidates - image + thoughts}


def _usage_value(usage: dict, key: str) -> float | None:
  value: object = usage
  for part in key.split("."):
    value = value.get(part) if isinstance(value, dict) else None
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return None
  return value


def _usd_from_rates(model: str, usage: dict) -> float | None:
  """Price ``usage`` from the rate table, or None.

  Every rated field must be present. A partial match would silently
  under-price the call, so it counts as unpriced.
  """
  rates = _load_rates().get(model)
  if not isinstance(rates, dict) or not rates:
    return None
  fields = {**usage, **_gemini_output_split(usage)}
  total = 0.0
  for kind, per_million in rates.items():
    tokens = _usage_value(fields, kind)
    if tokens is None:
      return None
    total += tokens / 1_000_000 * per_million
  return round(total, 6)


def attribution(meta: dict):
  """Context manager: every cost recorded inside the block carries `meta`
  (merged under the record's own meta). Lets callers attribute provider
  spend to a session/bird/operation without threading parameters through
  every image-API signature. Thread pools need `contextvars.copy_context()`
  to carry it into worker threads."""

  @contextlib.contextmanager
  def _ctx():
    token = _attribution_var.set({**(_attribution_var.get() or {}), **meta})
    try:
      yield
    finally:
      _attribution_var.reset(token)
  return _ctx()


def record(
  *,
  source: str,
  model: str,
  usage: dict | None,
  usd: float | None = None,
  meta: dict | None = None,
  request_id: str | None = None,
) -> None:
  """Append one call's usage to the ledger. Never raises — a metering
  failure must not fail the call being metered.

  ``usd`` is the provider's own cost figure when it reports one; otherwise
  the rate table is tried. ``request_id`` is the provider's id for the call
  (OpenRouter generation id, Anthropic message id, ...), which tells a
  retried or duplicated write apart from a genuine repeat call.
  """
  try:
    usage = usage or {}
    usd_source = "provider" if usd is not None else None
    if usd is None:
      usd = _usd_from_rates(model, usage)
      usd_source = "rates" if usd is not None else None
    row: dict = {
      "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
      "source": source,
      "model": model,
      "usage": usage,
      "usd": usd,
    }
    if usd_source:
      row["usd_source"] = usd_source
    if request_id:
      row["request_id"] = str(request_id)
    ambient = _attribution_var.get()
    merged = {**(ambient or {}), **(meta or {})}
    if merged:
      row["meta"] = merged
    # default=str: one Path or numpy value in meta must not drop the whole row.
    line = json.dumps(row, default=str)
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
      f.write(line + "\n")
  except Exception:
    pass


def _parse_ts(value: object) -> _dt.datetime | None:
  """Parse an ISO 8601 timestamp; naive values are UTC (the ledger's zone)."""
  if not isinstance(value, str):
    return None
  try:
    ts = _dt.datetime.fromisoformat(value)
  except ValueError:
    return None
  return ts if ts.tzinfo else ts.replace(tzinfo=_dt.timezone.utc)


def _row_usd_source(row: dict) -> str:
  return row.get("usd_source") or (
    "rates" if row.get("source") in _LEGACY_RATE_SOURCES else "provider"
  )


def summarize(since: str | None = None) -> dict:
  """Totals over the ledger, optionally from ``since`` (ISO 8601, any offset).

  ``usd_known`` = ``usd_metered`` (provider-stated) + ``usd_rates``
  (price-sheet estimate). ``usd_unknown_calls`` counts rows with no cost.
  """
  cutoff = _parse_ts(since) if since else None
  if since and cutoff is None:
    raise ValueError(f"since must be an ISO 8601 timestamp, got {since!r}")
  path = ledger_path()
  out: dict = {
    "calls": 0,
    "usd_known": 0.0,
    "usd_metered": 0.0,
    "usd_rates": 0.0,
    "usd_unknown_calls": 0,
    "by_model": {},
  }
  if not path.exists():
    return out
  for line in path.read_text().splitlines():
    try:
      row = json.loads(line)
    except json.JSONDecodeError:
      continue
    if not isinstance(row, dict):
      continue
    if cutoff is not None:
      ts = _parse_ts(row.get("ts"))
      if ts is None or ts < cutoff:
        continue
    out["calls"] += 1
    m = out["by_model"].setdefault(
      row.get("model", "?"), {"calls": 0, "usd": 0.0, "usd_rates": 0.0, "unknown": 0},
    )
    m["calls"] += 1
    usd = row.get("usd")
    if isinstance(usd, (int, float)) and not isinstance(usd, bool):
      bucket = "usd_rates" if _row_usd_source(row) == "rates" else "usd_metered"
      out["usd_known"] = round(out["usd_known"] + usd, 6)
      out[bucket] = round(out[bucket] + usd, 6)
      m["usd"] = round(m["usd"] + usd, 6)
      if bucket == "usd_rates":
        m["usd_rates"] = round(m["usd_rates"] + usd, 6)
    else:
      out["usd_unknown_calls"] += 1
      m["unknown"] += 1
  return out


if __name__ == "__main__":
  import argparse

  ap = argparse.ArgumentParser()
  ap.add_argument("--since", default=None, help="ISO 8601 lower bound, any UTC offset")
  args = ap.parse_args()
  try:
    summary = summarize(args.since)
  except ValueError as exc:
    ap.error(str(exc))
  print(json.dumps(summary, indent=2))
