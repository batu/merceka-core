"""Per-call provider cost ledger.

Every provider call that reports usage appends one JSONL row here. Token
counts come from the provider's own response (hard data, what billing is
computed from); `usd` is filled only when the provider states cost directly
(OpenRouter) or a rate table entry exists — never guessed.

Ledger path: $MERCEKA_COST_LEDGER, default ~/.merceka/costs.jsonl.
Summarize: `python -m merceka_core.costs [--since ISO8601]`.
"""

__all__ = ["record", "summarize", "ledger_path"]

import datetime as _dt
import json
import os
from pathlib import Path

# USD per 1M tokens by (model, token kind). Only entries verified against the
# provider's current price sheet belong here; unknown models simply get
# usd=None rows (tokens are still recorded, so cost is derivable later).
RATES_PATH_ENV = "MERCEKA_RATES_PATH"


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


import contextvars

_attribution_var: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
  "merceka_cost_attribution", default=None,
)


def _usd_from_rates(model: str, usage: dict) -> float | None:
  rates = _load_rates().get(model)
  if not isinstance(rates, dict):
    return None
  total = 0.0
  matched = False
  for kind, per_million in rates.items():
    tokens: object = usage
    for part in kind.split("."):
      tokens = tokens.get(part) if isinstance(tokens, dict) else None
    if isinstance(tokens, (int, float)):
      total += tokens / 1_000_000 * per_million
      matched = True
  return round(total, 6) if matched else None


def attribution(meta: dict):
  """Context manager: every cost recorded inside the block carries `meta`
  (merged under the record's own meta). Lets callers attribute provider
  spend to a session/bird/operation without threading parameters through
  every image-API signature."""
  import contextlib

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
) -> None:
  """Append one call's usage to the ledger. Never raises — a metering
  failure must not fail the call being metered."""
  try:
    usage = usage or {}
    if usd is None:
      usd = _usd_from_rates(model, usage)
    row = {
      "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
      "source": source,
      "model": model,
      "usage": usage,
      "usd": usd,
    }
    ambient = _attribution_var.get()
    merged = {**(ambient or {}), **(meta or {})}
    if merged:
      row["meta"] = merged
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
      f.write(json.dumps(row) + "\n")
  except Exception:
    pass


def summarize(since: str | None = None) -> dict:
  path = ledger_path()
  out: dict = {"calls": 0, "usd_known": 0.0, "usd_unknown_calls": 0, "by_model": {}}
  if not path.exists():
    return out
  for line in path.read_text().splitlines():
    try:
      row = json.loads(line)
    except json.JSONDecodeError:
      continue
    if since and row.get("ts", "") < since:
      continue
    out["calls"] += 1
    m = out["by_model"].setdefault(row.get("model", "?"), {"calls": 0, "usd": 0.0, "unknown": 0})
    m["calls"] += 1
    usd = row.get("usd")
    if isinstance(usd, (int, float)):
      out["usd_known"] = round(out["usd_known"] + usd, 6)
      m["usd"] = round(m["usd"] + usd, 6)
    else:
      out["usd_unknown_calls"] += 1
      m["unknown"] += 1
  return out


if __name__ == "__main__":
  import argparse

  ap = argparse.ArgumentParser()
  ap.add_argument("--since", default=None, help="ISO8601 lower bound")
  args = ap.parse_args()
  print(json.dumps(summarize(args.since), indent=2))
