"""Multi-model OpenRouter vision critique panel.

This module is intentionally self-contained so multiple studio tools can share
one deterministic parser and aggregation contract while still swapping judge
registries or transports in tests.
"""

from __future__ import annotations

import base64
import inspect
import json
import math
import mimetypes
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Callable

import httpx

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"
_ENV_FALLBACK_PATH = Path("/Users/base/dev/appletolye/.env")

JUDGE_REGISTRY: list[dict[str, Any]] = [
  {
    "id": "anthropic/claude-opus-4.8",
    "model": "anthropic/claude-opus-4.8",
    "enabled": True,
  },
  {
    "id": "anthropic/claude-sonnet-4.6",
    "model": "anthropic/claude-sonnet-4.6",
    "enabled": True,
  },
  {
    "id": "google/gemini-3.5-flash",
    "model": "google/gemini-3.5-flash",
    "enabled": True,
  },
  {
    "id": "openai/gpt-5",
    "model": "openai/gpt-5",
    "enabled": False,
  },
]

FINDING_KEYS = [
  "layout",
  "color",
  "typography",
  "text-content",
  "missing-element",
  "extra-element",
  "sizing",
  "spacing",
  "iconography",
  "background",
  "other",
]
SEVERITY_RANK = {"blocker": 3, "major": 2, "minor": 1}
_SEVERITIES = set(SEVERITY_RANK)
_MAX_TEXT = 400

CRITIQUE_PROMPT = f"""You are the studio's shared multi-model visual fidelity judge.
Compare the supplied image(s) against the target.

If a reference image is supplied, IMAGE 1 is the REFERENCE target and each
following image is OURS. If no reference image is supplied, judge OURS against
the written spec. List ranked visual differences that make OURS deviate from
the target.

Respond with ONLY a JSON object, no prose, in this exact shape:
{{
  "score": <number 0-100, visual fidelity percent>,
  "defects": [
    {{
      "key": <one of: {", ".join(FINDING_KEYS)}>,
      "region": "<short location, include image number if multiple OURS images>",
      "severity": <one of: blocker, major, minor>,
      "defect": "<short: what differs>",
      "direction": "<short: how OURS should move toward the target>"
    }}
  ]
}}

Order defects most-severe first. Use "blocker" only for differences that break
the screen's identity or usability. If the image(s) match the target, return an
empty defects array."""

_OPENROUTER_RESPONSE_FORMAT = {
  "type": "json_schema",
  "json_schema": {
    "name": "vision_critique",
    "strict": True,
    "schema": {
      "type": "object",
      "additionalProperties": False,
      "required": ["score", "defects"],
      "properties": {
        "score": {"type": "number"},
        "defects": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["key", "region", "severity", "defect", "direction"],
            "properties": {
              "key": {"type": "string", "enum": FINDING_KEYS},
              "region": {"type": "string"},
              "severity": {"type": "string", "enum": ["blocker", "major", "minor"]},
              "defect": {"type": "string"},
              "direction": {"type": "string"},
            },
          },
        },
      },
    },
  },
}


def critique(
  images: list[str | Path | bytes | bytearray | memoryview],
  reference: str | Path | None = None,
  spec: str | None = None,
  judges: list[str | dict[str, Any]] | None = None,
  budget_check: Callable[..., Any] | None = None,
  *,
  floor: float = 85.0,
  timeout: float = 60.0,
  client: httpx.Client | None = None,
) -> dict[str, Any]:
  """Run a multi-model visual critique and aggregate participating judges.

  Args:
    images: One or more image paths or raw image bytes to judge.
    reference: Optional reference image path. When present, judges compare OURS
      against this target; otherwise they use ``spec`` as the target.
    spec: Optional written target/specification.
    judges: Optional judge roster. Items may be strings or dicts with
      ``id``, ``model``, and optional ``enabled``.
    budget_check: Optional callable run before each billable judge call. A
      falsy return skips the current and remaining judges with reason
      ``"budget"``.
    floor: Informational pass/fail score floor. Defaults to 85.
    timeout: Per-judge HTTP timeout in seconds when this function owns the
      client.
    client: Optional injected ``httpx.Client`` for tests or caller-managed
      connection reuse.

  Returns:
    A dict with score, verdict, defects, per_model, consensus, participated,
    and skipped.

  Raises:
    RuntimeError: When no judge produced a parseable score.
  """
  if not images:
    raise ValueError("critique requires at least one image")

  api_key = _openrouter_api_key()
  if not api_key:
    raise RuntimeError(
      f"OPENROUTER_API_KEY is not configured in the environment or {_ENV_FALLBACK_PATH}"
    )

  roster = _normalize_judges(judges)
  messages = _build_messages(images, reference=reference, spec=spec)
  per_model: dict[str, dict[str, Any]] = {}
  participated: list[str] = []
  skipped: list[dict[str, str]] = []
  participant_results: list[dict[str, Any]] = []

  owns_client = client is None
  http_client = client or httpx.Client(timeout=timeout)
  budget_halted = False

  try:
    for judge in roster:
      judge_id = judge["id"]
      if not judge.get("enabled", True):
        _record_skip(per_model, skipped, judge, "disabled")
        continue
      if budget_halted:
        _record_skip(per_model, skipped, judge, "budget")
        continue
      if budget_check is not None and not _budget_allows(budget_check, judge):
        budget_halted = True
        _record_skip(per_model, skipped, judge, "budget")
        continue

      result = _call_judge(http_client, judge, messages, api_key)
      if result["ok"]:
        per_model[judge_id] = {
          "model": judge["model"],
          "score": result["score"],
          "defects": [_public_defect(d, include_key=True) for d in result["defects"]],
        }
        participated.append(judge_id)
        participant_results.append(
          {
            "judge": judge_id,
            "model": judge["model"],
            "score": result["score"],
            "defects": result["defects"],
          }
        )
      else:
        _record_skip(per_model, skipped, judge, result["reason"])
  finally:
    if owns_client:
      http_client.close()

  if not participant_results:
    reasons = ", ".join(f"{s['judge']}: {s['reason']}" for s in skipped) or "none"
    raise RuntimeError(f"vision critique had 0 participating judges; skipped={reasons}")

  score = float(median([r["score"] for r in participant_results]))
  consensus = _consensus_keys(participant_results)
  defects = [_public_defect(d) for d in _aggregate_defects(participant_results)]
  consensus_blocker = _has_consensus_blocker(consensus, participant_results)
  verdict = "fail" if score < floor or consensus_blocker else "pass"

  return {
    "score": score,
    "verdict": verdict,
    "defects": defects,
    "per_model": per_model,
    "consensus": consensus,
    "participated": participated,
    "skipped": skipped,
  }


def openrouter_budget_floor(
  floor_usd: float = 5.0,
  *,
  api_key: str | None = None,
  client: httpx.Client | None = None,
  timeout: float = 15.0,
) -> Callable[[], bool]:
  """Return a budget guard that checks OpenRouter remaining credits.

  The credits endpoint returns total credits and total usage; remaining balance
  is computed as ``total_credits - total_usage``.
  """

  def check() -> bool:
    key = api_key or _openrouter_api_key()
    if not key:
      return False
    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout)
    try:
      response = http_client.get(
        OPENROUTER_CREDITS_URL,
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
      )
      if response.status_code != 200:
        return False
      body = response.json()
      if not isinstance(body, dict):
        return False
      data = body.get("data", {})
      if not isinstance(data, dict):
        return False
      total_credits = float(data.get("total_credits", 0.0))
      total_usage = float(data.get("total_usage", 0.0))
      return total_credits - total_usage >= floor_usd
    except (TypeError, ValueError, httpx.HTTPError, json.JSONDecodeError):
      return False
    finally:
      if owns_client:
        http_client.close()

  return check


def _call_judge(
  client: httpx.Client,
  judge: dict[str, Any],
  messages: list[dict[str, Any]],
  api_key: str,
) -> dict[str, Any]:
  payload = {
    "model": judge["model"],
    "messages": messages,
    "temperature": 0,
    # response_format is best-effort: providers that ignore it (e.g. anthropic)
    # fall back to the tolerant fenced/prose parser. require_parameters would
    # hard-400 those providers (found live: anthropic 400 vs gemini 200).
    "response_format": _OPENROUTER_RESPONSE_FORMAT,
  }
  try:
    response = client.post(
      OPENROUTER_CHAT_URL,
      json=payload,
      headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
      },
    )
  except httpx.TimeoutException:
    return {"ok": False, "reason": "timeout"}
  except httpx.HTTPError:
    return {"ok": False, "reason": "request-error"}

  if response.status_code != 200:
    return {"ok": False, "reason": _skip_reason_for_status(response.status_code)}

  try:
    content = _extract_openrouter_text(response.json())
    parsed = parse_judge_response(content)
  except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    return {"ok": False, "reason": "parse-failure"}

  return {"ok": True, **parsed}


def parse_judge_response(text: str) -> dict[str, Any]:
  """Parse one judge response into a clamped score and normalized defects."""
  if not isinstance(text, str):
    raise ValueError("judge response is not text")
  obj = _extract_json_object(text)
  if obj is None:
    return _parse_prose_response(text)

  raw_score = obj.get("score", obj.get("fidelity"))
  score = _clamp_score(raw_score)
  if score is None:
    raise ValueError("judge response missing numeric score")

  raw_defects = obj.get("defects", obj.get("findings", []))
  defects = [_normalize_defect(d) for d in raw_defects] if isinstance(raw_defects, list) else []
  defects.sort(key=lambda d: SEVERITY_RANK[d["severity"]], reverse=True)
  return {"score": score, "defects": defects}


def _extract_json_object(text: str) -> dict[str, Any] | None:
  start = text.find("{")
  end = text.rfind("}")
  if start == -1 or end <= start:
    return None
  try:
    return json.loads(text[start : end + 1])
  except json.JSONDecodeError:
    return None


def _parse_prose_response(text: str) -> dict[str, Any]:
  match = re.search(r"\b(?:score|fidelity)\b[^0-9-]*(-?\d+(?:\.\d+)?)\s*%?", text, re.I)
  if match is None:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*%\s*(?:fidelity|match|score)", text, re.I)
  score = _clamp_score(match.group(1) if match else None)
  if score is None:
    raise ValueError("judge prose response missing numeric score")

  defects = []
  for line in text.splitlines():
    sev = re.search(r"\b(blocker|major|minor)\b", line, re.I)
    if not sev:
      continue
    severity = sev.group(1).lower()
    key_match = re.search(
      r"\b(" + "|".join(re.escape(k) for k in FINDING_KEYS) + r")\b", line, re.I
    )
    key = _normalize_key(key_match.group(1).lower() if key_match else line)
    defect_text = re.sub(r"^\s*[-*0-9.)\s]+", "", line).strip()
    defects.append(
      {
        "key": key,
        "region": "unspecified",
        "severity": severity,
        "defect": _clip(defect_text),
        "direction": _clip(defect_text),
      }
    )

  defects.sort(key=lambda d: SEVERITY_RANK[d["severity"]], reverse=True)
  return {"score": score, "defects": defects}


def _normalize_defect(raw: Any) -> dict[str, str]:
  if not isinstance(raw, dict):
    raw = {"defect": str(raw)}

  defect_text = (
    raw.get("defect") or raw.get("description") or raw.get("issue") or raw.get("key") or ""
  )
  direction = raw.get("direction")
  if not direction:
    reference = raw.get("reference")
    ours = raw.get("ours")
    if reference or ours:
      direction = f"reference: {reference or ''}; ours: {ours or ''}"
  key = _normalize_key(raw.get("key") or defect_text)
  severity = str(raw.get("severity", "minor")).lower()
  if severity not in _SEVERITIES:
    severity = "minor"

  return {
    "key": key,
    "region": _clip(raw.get("region") or raw.get("location") or raw.get("area") or "unspecified"),
    "severity": severity,
    "defect": _clip(defect_text),
    "direction": _clip(direction or ""),
  }


def _normalize_key(value: Any) -> str:
  if isinstance(value, str):
    value = value.strip().lower()
    if value in FINDING_KEYS:
      return value
    compact = re.sub(r"[\s_]+", "-", value)
    if compact in FINDING_KEYS:
      return compact
    aliases = {
      "text": "text-content",
      "copy": "text-content",
      "font": "typography",
      "fonts": "typography",
      "type": "typography",
      "size": "sizing",
      "alignment": "layout",
      "position": "layout",
      "missing": "missing-element",
      "extra": "extra-element",
      "icon": "iconography",
      "icons": "iconography",
      "bg": "background",
    }
    words = set(re.findall(r"[a-z0-9]+", value))
    for word, key in aliases.items():
      if word in words:
        return key
  return "other"


def _clamp_score(value: Any) -> float | None:
  try:
    score = float(value)
  except (TypeError, ValueError):
    return None
  if not math.isfinite(score):
    return None
  return max(0.0, min(100.0, score))


def _extract_openrouter_text(body: dict[str, Any]) -> str:
  content = body["choices"][0]["message"]["content"]
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    text_parts = []
    for part in content:
      if isinstance(part, dict) and part.get("type") == "text":
        text_parts.append(str(part.get("text", "")))
    return "\n".join(text_parts)
  raise ValueError("OpenRouter response content is not text")


def _build_messages(
  images: list[str | Path | bytes | bytearray | memoryview],
  *,
  reference: str | Path | None,
  spec: str | None,
) -> list[dict[str, Any]]:
  content: list[dict[str, Any]] = [{"type": "text", "text": _prompt_with_spec(spec, reference)}]
  if reference is not None:
    content.append({"type": "text", "text": "IMAGE 1: REFERENCE"})
    content.append(_image_url_part(reference))
  for index, image in enumerate(images, start=1):
    label = f"OURS {index}" if reference is not None else f"IMAGE {index}: OURS"
    content.append({"type": "text", "text": label})
    content.append(_image_url_part(image))
  return [{"role": "user", "content": content}]


def _prompt_with_spec(spec: str | None, reference: str | Path | None) -> str:
  if spec:
    return f"{CRITIQUE_PROMPT}\n\nWritten spec:\n{spec}"
  if reference is None:
    return (
      f"{CRITIQUE_PROMPT}\n\nNo reference image or written spec was supplied; "
      "judge internal visual quality and report only concrete defects."
    )
  return CRITIQUE_PROMPT


def _image_url_part(image: str | Path | bytes | bytearray | memoryview) -> dict[str, Any]:
  data, mime_type = _read_image_bytes(image)
  encoded = base64.b64encode(data).decode("ascii")
  return {
    "type": "image_url",
    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
  }


def _read_image_bytes(image: str | Path | bytes | bytearray | memoryview) -> tuple[bytes, str]:
  if isinstance(image, (bytes, bytearray, memoryview)):
    data = bytes(image)
    return data, _mime_from_bytes(data)
  path = Path(image)
  data = path.read_bytes()
  guessed, _ = mimetypes.guess_type(str(path))
  return data, guessed or _mime_from_bytes(data)


def _mime_from_bytes(data: bytes) -> str:
  if data.startswith(b"\x89PNG\r\n\x1a\n"):
    return "image/png"
  if data.startswith(b"\xff\xd8\xff"):
    return "image/jpeg"
  if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
    return "image/gif"
  if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
    return "image/webp"
  return "image/png"


def _normalize_judges(judges: list[str | dict[str, Any]] | None) -> list[dict[str, Any]]:
  source = judges if judges is not None else [j for j in JUDGE_REGISTRY if j.get("enabled", True)]
  normalized = []
  registry_by_id = {j["id"]: j for j in JUDGE_REGISTRY}
  for item in source:
    if isinstance(item, str):
      registry_item = registry_by_id.get(item)
      if registry_item and judges is None:
        normalized.append(dict(registry_item))
      else:
        normalized.append(
          {
            "id": item,
            "model": registry_item.get("model", item) if registry_item else item,
            "enabled": True,
          }
        )
      continue

    model = item.get("model") or item.get("id")
    judge_id = item.get("id") or model
    if not model or not judge_id:
      raise ValueError(f"invalid judge registry item: {item!r}")
    normalized.append(
      {
        "id": str(judge_id),
        "model": str(model),
        "enabled": bool(item.get("enabled", True)),
      }
    )
  return normalized


def _budget_allows(fn: Callable[..., Any], judge: dict[str, Any]) -> bool:
  context = {"judge": judge["id"], "model": judge["model"]}
  try:
    signature = inspect.signature(fn)
  except (TypeError, ValueError):
    return bool(fn())

  accepts_positional = any(
    p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
    for p in signature.parameters.values()
  )
  if accepts_positional:
    return bool(fn(context))
  return bool(fn())


def _record_skip(
  per_model: dict[str, dict[str, Any]],
  skipped: list[dict[str, str]],
  judge: dict[str, Any],
  reason: str,
) -> None:
  per_model[judge["id"]] = {"model": judge["model"], "skipped": True, "reason": reason}
  skipped.append({"judge": judge["id"], "reason": reason})


def _skip_reason_for_status(status_code: int) -> str:
  return f"HTTP {status_code}"


def _consensus_keys(results: list[dict[str, Any]]) -> list[str]:
  threshold = math.ceil(len(results) / 2)
  counts: Counter[str] = Counter()
  for result in results:
    counts.update({d["key"] for d in result["defects"]})
  return [
    key
    for key, _count in sorted(
      ((key, count) for key, count in counts.items() if count >= threshold),
      key=lambda item: (-item[1], FINDING_KEYS.index(item[0]) if item[0] in FINDING_KEYS else 999),
    )
  ]


def _aggregate_defects(results: list[dict[str, Any]]) -> list[dict[str, str]]:
  by_signature: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
  for result in results:
    seen = set()
    for defect in result["defects"]:
      signature = (defect["key"], defect["region"])
      if signature in seen:
        continue
      seen.add(signature)
      by_signature[signature].append(defect)

  aggregated = []
  for (key, _region), defects in by_signature.items():
    defects = sorted(defects, key=lambda d: SEVERITY_RANK[d["severity"]], reverse=True)
    representative = dict(defects[0])
    representative["key"] = key
    aggregated.append(representative)

  aggregated.sort(key=lambda d: (SEVERITY_RANK[d["severity"]], d["key"]), reverse=True)
  return aggregated


def _has_consensus_blocker(consensus: list[str], results: list[dict[str, Any]]) -> bool:
  threshold = math.ceil(len(results) / 2)
  for key in consensus:
    blocker_votes = 0
    for result in results:
      for defect in result["defects"]:
        if defect["key"] == key and defect["severity"] == "blocker":
          blocker_votes += 1
          break
    if blocker_votes >= threshold:
      return True
  return False


def _public_defect(defect: dict[str, str], *, include_key: bool = False) -> dict[str, str]:
  public = {
    "region": defect["region"],
    "severity": defect["severity"],
    "defect": defect["defect"],
    "direction": defect["direction"],
  }
  if include_key:
    return {"key": defect["key"], **public}
  return public


def _clip(value: Any) -> str:
  return str(value or "")[:_MAX_TEXT]


def _openrouter_api_key() -> str | None:
  key = os.getenv("OPENROUTER_API_KEY")
  if key:
    return key
  return _env_file_key(_ENV_FALLBACK_PATH)


def _env_file_key(path: Path) -> str | None:
  try:
    lines = path.read_text().splitlines()
  except OSError:
    return None
  for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
      continue
    name, value = stripped.split("=", 1)
    name = name.removeprefix("export ").strip()
    if name == "OPENROUTER_API_KEY":
      return value.strip().strip("\"'") or None
  return None
