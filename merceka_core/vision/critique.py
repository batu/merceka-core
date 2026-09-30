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
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Callable

import httpx
import shutil

from merceka_core import _cli, _env
from merceka_core import costs as _costs
from merceka_core.vision import zoom_judge as _zoom_judge
import subprocess
import tempfile as _tempfile

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"

JUDGE_REGISTRY: list[dict[str, Any]] = [
  {
    # Runs through the local `codex` CLI (subscription billing), not OpenRouter.
    "id": "codex/gpt-5.6-terra",
    "model": "gpt-5.6-terra",
    "cli": "codex",
    "effort": "max",
    "enabled": True,
  },
  {
    # Agentic zoom judge: bills the Anthropic API directly (key-gated on
    # ANTHROPIC_API_KEY), magnifies regions before judging small detail.
    "id": "anthropic/claude-fable-5-zoom",
    "model": "claude-fable-5",
    "api": "anthropic-zoom",
    # Off by default: bills the Anthropic API directly rather than a
    # subscription. Re-enable per-call via an explicit roster when a job needs
    # magnified small-detail judging.
    "enabled": False,
  },
  {
    # Runs through the local `claude` CLI (subscription billing), not OpenRouter.
    "id": "anthropic/claude-fable-5",
    "model": "claude-fable-5",
    "cli": "claude",
    "enabled": True,
  },
  {
    # Runs through the local `claude` CLI (subscription billing), not OpenRouter.
    "id": "anthropic/claude-opus-5",
    "model": "claude-opus-5",
    "cli": "claude",
    "enabled": True,
  },
  {
    "id": "anthropic/claude-opus-4.8",
    "model": "anthropic/claude-opus-4.8",
    "enabled": False,
  },
  {
    "id": "anthropic/claude-sonnet-4.6",
    "model": "anthropic/claude-sonnet-4.6",
    "enabled": False,
  },
  {
    # The only judge that spends OpenRouter credits.
    "id": "google/gemini-3.6-flash",
    "model": "google/gemini-3.6-flash",
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
# Severity words models use, matched case- and whitespace-insensitively. Anything
# else is minor.
_SEVERITY_ALIASES = {
  "blocker": "blocker",
  "critical": "blocker",
  "high": "blocker",
  "major": "major",
  "medium": "major",
  "minor": "minor",
  "low": "minor",
}
_MAX_TEXT = 400
RECURRING_CHECKS = [
  {
    "id": "banner-transparency",
    "description": (
      "sprite banners/ribbons must composite transparently over the background; "
      "an opaque box behind one fails"
    ),
  },
  {
    "id": "asset-identity",
    "description": (
      "an element visibly using a different asset than the reference's, including "
      "a font glyph standing in for art, fails"
    ),
  },
  {
    "id": "glyph-centering",
    "description": (
      "glyphs in circular/pill buttons, such as a '+', must be visually centered"
    ),
  },
  {
    "id": "content-containment",
    "description": "text/icons must not leak outside their pill/chip/button bounds",
  },
  {
    "id": "sibling-size-consistency",
    "description": "cells/buttons in one row must be equal-sized",
  },
]
RECURRING_CHECK_IDS = [check["id"] for check in RECURRING_CHECKS]
_RECURRING_CHECK_ID_SET = set(RECURRING_CHECK_IDS)
_RECURRING_CHECK_PROMPT = "\n".join(
  f"- {check['id']}: {check['description']}" for check in RECURRING_CHECKS
)

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
  ],
  "recurring_checks": [
    {{
      "id": <one of: {", ".join(RECURRING_CHECK_IDS)}>,
      "subject": "<the recurring check subject this entry judges, exactly as listed>",
      "pass": <true if this recurring defect is absent, false if present>,
      "evidence": "<short phrase naming the subject and pixel location>"
    }}
  ]
}}

Order defects most-severe first. Use "blocker" only for differences that break
the screen's identity or usability. If the image(s) match the target, return an
empty defects array.

Recurring defects checklist:
{_RECURRING_CHECK_PROMPT}

For recurring_checks, emit one entry per judged subject per checklist id, with
subject set to that subject's name exactly as listed under "Recurring check
subjects". If multiple OURS images or named crop subjects are listed, repeat the
ids for each subject and include the image/crop name plus a pixel location in
evidence."""

_OPENROUTER_RESPONSE_FORMAT = {
  "type": "json_schema",
  "json_schema": {
    "name": "vision_critique",
    "strict": True,
    "schema": {
      "type": "object",
      "additionalProperties": False,
      "required": ["score", "defects", "recurring_checks"],
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
        "recurring_checks": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "subject", "pass", "evidence"],
            "properties": {
              "id": {"type": "string", "enum": RECURRING_CHECK_IDS},
              "subject": {"type": "string"},
              "pass": {"type": "boolean"},
              "evidence": {"type": "string"},
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
  recurring_check_units: list[str] | None = None,
  floor: float = 85.0,
  timeout: float = 60.0,
  client: httpx.Client | None = None,
  quorum: int | None = None,
  recurring_checks_gate: bool = True,
) -> dict[str, Any]:
  """Run a multi-model visual critique and aggregate participating judges.

  The verdict is ``"fail"`` when the median score is below ``floor``, when a
  majority of participants flag a blocker under the same defect key, or (with
  ``recurring_checks_gate``) when a majority of participants fail the same
  recurring check for the same subject. Majority means ``ceil(n / 2)`` of the n
  participating judges. Defect consensus matches on the finding key only: judges
  that report the same blocker under different keys (say "background" and
  "extra-element" for one opaque box) do not agree.

  Args:
    images: One or more image paths or raw image bytes to judge.
    reference: Optional reference image path. When present, judges compare OURS
      against this target; otherwise they use ``spec`` as the target.
    spec: Optional written target/specification.
    judges: Optional judge roster. Items may be strings or dicts with
      ``id``, ``model``, and optional ``enabled``. A registry id, given as a
      string or a partial dict, keeps the registry's model and transport
      (``cli``, ``effort``, ``api``); fields set in the dict override them.
      Listing a judge enables it unless its dict sets ``enabled: False``.
    budget_check: Optional callable run before each paid API judge call
      (OpenRouter and the Anthropic zoom judge); CLI judges run on
      subscriptions and are never gated. It may take no arguments or one
      context dict with ``judge``, ``model`` and ``transport``
      (``"openrouter"`` or ``"anthropic"``). A falsy return skips that judge
      and every later paid judge with reason ``"budget"``.
    recurring_check_units: Optional labels for the subjects that need recurring
      defect checks. Defaults to one subject per OURS image.
    floor: Informational pass/fail score floor. Defaults to 85.
    timeout: Per-judge HTTP timeout in seconds when this function owns the
      client.
    client: Optional injected ``httpx.Client`` for tests or caller-managed
      connection reuse.
    quorum: Minimum number of judges that must return a parseable score.
      Defaults to a majority of the enabled roster, ``ceil(n / 2)`` and at least
      1, so a panel that lost most of its judges (expired CLI login, HTTP
      errors, parse failures) raises instead of letting the survivors decide
      alone. Pass ``quorum=1`` to accept any non-empty panel.
    recurring_checks_gate: When True (the default), a recurring check that a
      majority of participants fail for the same subject fails the verdict.
      These checks cover defects that keep recurring, so they gate like a
      consensus blocker. Pass False to restore the score-and-blocker verdict;
      ``failed_recurring_checks`` is reported either way.

  Returns:
    A dict with score, verdict, defects, per_model, consensus, participated,
    and skipped, plus ``participants`` (judges that returned a score),
    ``roster_size`` (enabled judges) and ``degraded`` (True when any enabled
    judge was skipped). Judges disabled in the roster are listed in skipped
    but do not count toward the roster or degrade the panel. A paid judge
    whose key (``OPENROUTER_API_KEY`` or ``ANTHROPIC_API_KEY``) is not
    configured is skipped with reason ``"no-key"`` and counts against the
    quorum.
    ``recurring_checks`` shows, per subject and check, a failure if any judge
    reported one; ``failed_recurring_checks`` lists the check ids that failed
    by majority.

  Raises:
    RuntimeError: When fewer than ``quorum`` judges produced a parseable score.
      The message names every skipped judge and its reason.
    ValueError: When ``images`` is empty or ``quorum`` is below 1 or above the
      number of enabled judges.
  """
  if not images:
    raise ValueError("critique requires at least one image")

  roster = _normalize_judges(judges)
  roster_size = sum(1 for judge in roster if judge.get("enabled", True))
  if quorum is not None and not 1 <= quorum <= roster_size:
    raise ValueError(
      f"quorum {quorum} exceeds the {roster_size} enabled judges"
      if quorum > roster_size
      else f"quorum must be at least 1, got {quorum}"
    )
  required = quorum if quorum is not None else max(1, math.ceil(roster_size / 2))
  check_units = _normalize_recurring_check_units(images, recurring_check_units)
  messages = _build_messages(
    images,
    reference=reference,
    spec=spec,
    recurring_check_units=check_units,
  )
  per_model: dict[str, dict[str, Any]] = {}
  participated: list[str] = []
  skipped: list[dict[str, str]] = []
  participant_results: list[dict[str, Any]] = []

  owns_client = client is None
  http_client = client or httpx.Client(timeout=timeout)
  budget_halted = False
  # Resolved once, only if an OpenRouter judge runs; "" means no key.
  openrouter_key: str | None = None

  try:
    for judge in roster:
      judge_id = judge["id"]
      if not judge.get("enabled", True):
        _record_skip(per_model, skipped, judge, "disabled")
        continue
      if judge.get("cli") == "codex":
        # Bills the codex subscription, so neither the budget gate nor any
        # API key applies.
        result = _call_codex_cli_judge(judge, images, reference, spec, check_units)
      elif judge.get("cli") == "claude":
        # Bills the Claude subscription through the local CLI.
        result = _call_claude_cli_judge(judge, images, reference, spec, check_units)
      elif judge.get("api") == "anthropic-zoom":
        # Bills the Anthropic API directly.
        anthropic_key = _anthropic_api_key()
        if not anthropic_key:
          _record_skip(per_model, skipped, judge, "no-key")
          continue
        if budget_halted or not _budget_allows(budget_check, judge, "anthropic"):
          budget_halted = True
          _record_skip(per_model, skipped, judge, "budget")
          continue
        result = _call_anthropic_zoom_judge(
          judge, images, reference, spec, check_units, api_key=anthropic_key
        )
      else:
        # Spends OpenRouter credits.
        if openrouter_key is None:
          openrouter_key = _openrouter_api_key() or ""
        if not openrouter_key:
          _record_skip(per_model, skipped, judge, "no-key")
          continue
        if budget_halted or not _budget_allows(budget_check, judge, "openrouter"):
          budget_halted = True
          _record_skip(per_model, skipped, judge, "budget")
          continue
        result = _call_judge(http_client, judge, messages, openrouter_key, check_units)
      if result["ok"]:
        per_model[judge_id] = {
          "model": judge["model"],
          "score": result["score"],
          "defects": [_public_defect(d, include_key=True) for d in result["defects"]],
          "recurring_checks": [_public_recurring_check(c) for c in result["recurring_checks"]],
        }
        participated.append(judge_id)
        participant_results.append(
          {
            "judge": judge_id,
            "model": judge["model"],
            "score": result["score"],
            "defects": result["defects"],
            "recurring_checks": result["recurring_checks"],
          }
        )
      else:
        _record_skip(per_model, skipped, judge, result["reason"])
  finally:
    if owns_client:
      http_client.close()

  participants = len(participant_results)
  if participants < required:
    reasons = ", ".join(f"{s['judge']}: {s['reason']}" for s in skipped) or "none"
    raise RuntimeError(
      f"vision critique had {participants} participating judges, below the quorum of "
      f"{required} (roster of {roster_size}); skipped={reasons}"
    )

  score = float(median([r["score"] for r in participant_results]))
  consensus = _consensus_keys(participant_results)
  defects = [_public_defect(d) for d in _aggregate_defects(participant_results)]
  recurring_checks = [
    _public_recurring_check(c) for c in _aggregate_recurring_checks(participant_results, check_units)
  ]
  consensus_blocker = _has_consensus_blocker(consensus, participant_results)
  failed_recurring_checks = _failed_recurring_checks(participant_results)
  recurring_fail = recurring_checks_gate and bool(failed_recurring_checks)
  verdict = "fail" if score < floor or consensus_blocker or recurring_fail else "pass"

  return {
    "score": score,
    "verdict": verdict,
    "defects": defects,
    "recurring_checks": recurring_checks,
    "failed_recurring_checks": failed_recurring_checks,
    "per_model": per_model,
    "consensus": consensus,
    "participated": participated,
    "skipped": skipped,
    "participants": participants,
    "roster_size": roster_size,
    "degraded": participants < roster_size,
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


def _call_anthropic_zoom_judge(
  judge: dict[str, Any],
  images: list[str | Path | bytes | bytearray | memoryview],
  reference: str | Path | None,
  spec: str | None,
  recurring_check_units: list[str],
  *,
  api_key: str,
) -> dict[str, Any]:
  raw = _zoom_judge.call_zoom_judge(
    judge,
    images,
    reference,
    _prompt_with_spec(spec, reference, recurring_check_units),
    api_key=api_key,
  )
  if not raw["ok"]:
    return raw
  try:
    parsed = parse_judge_response(raw["text"], recurring_check_units=recurring_check_units)
  except (TypeError, ValueError, json.JSONDecodeError):
    return {"ok": False, "reason": "parse-failure"}
  return {"ok": True, **parsed}


def _call_codex_cli_judge(
  judge: dict[str, Any],
  images: list[str | Path | bytes | bytearray | memoryview],
  reference: str | Path | None,
  spec: str | None,
  recurring_check_units: list[str],
) -> dict[str, Any]:
  """Run one judge through the local `codex` CLI (vision via -i attachments).

  The judge runs read-only in a scratch working directory, so the calling
  repo's AGENTS.md never reaches it, and its answer is read from
  ``--output-last-message`` instead of being cut out of the stdout transcript.
  """
  binary = shutil.which("codex") or "/opt/homebrew/bin/codex"
  prompt = _prompt_with_spec(spec, reference, recurring_check_units)
  ordered: list[str | Path | bytes | bytearray | memoryview] = []
  if reference is not None:
    prompt += "\n\nAttached images, in order: image 1 is the REFERENCE; the rest are OURS."
    ordered.append(reference)
  else:
    prompt += "\n\nAttached images are OURS, in order."
  ordered.extend(images)

  try:
    with _tempfile.TemporaryDirectory(prefix="critique-codex-") as workdir:
      attachments: list[str] = []
      for index, item in enumerate(ordered):
        if isinstance(item, (bytes, bytearray, memoryview)):
          path = Path(workdir, f"image-{index}.png")
          path.write_bytes(bytes(item))
          attachments.append(str(path))
        else:
          # Relative paths would otherwise resolve against the scratch cwd.
          attachments.append(str(Path(item).resolve()))
      last_message = Path(workdir, "last-message.txt")
      cmd = _cli.codex_exec_command(judge["model"], cd=workdir, images=attachments, binary=binary)
      # codex_exec_command applies reasoning effort only to alias models; the
      # judge names both a model and an effort.
      cmd[2:2] = [
        "-c", f'model_reasoning_effort="{judge.get("effort", "high")}"',
        "--output-last-message", str(last_message),
      ]
      completed = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, timeout=600,
        cwd=workdir, env=_cli.codex_env(),
      )
      if completed.returncode != 0:
        return {"ok": False, "reason": "cli-error"}
      text = last_message.read_text() if last_message.exists() else ""
  except (OSError, subprocess.TimeoutExpired):
    return {"ok": False, "reason": "cli-error"}

  try:
    parsed = parse_judge_response(text, recurring_check_units=recurring_check_units)
  except (TypeError, ValueError, json.JSONDecodeError):
    return {"ok": False, "reason": "parse-failure"}
  return {"ok": True, **parsed}


def _call_claude_cli_judge(
  judge: dict[str, Any],
  images: list[str | Path | bytes | bytearray | memoryview],
  reference: str | Path | None,
  spec: str | None,
  recurring_check_units: list[str],
) -> dict[str, Any]:
  """Run one judge through the local `claude` CLI on subscription billing.

  Unlike `codex exec -i`, `claude -p` has no image-attachment flag, so images
  are named by absolute path and read with the Read tool. The judge runs in a
  scratch cwd so the calling repo's CLAUDE.md and hooks never leak into the
  critique prompt.
  """
  binary = shutil.which("claude") or "/opt/homebrew/bin/claude"
  prompt = _prompt_with_spec(spec, reference, recurring_check_units)
  ordered: list[str | Path | bytes | bytearray | memoryview] = []
  if reference is not None:
    ordered.append(reference)
  ordered.extend(images)

  temps: list[str] = []
  try:
    paths: list[Path] = []
    for item in ordered:
      if isinstance(item, (bytes, bytearray, memoryview)):
        handle = _tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        handle.write(bytes(item))
        handle.close()
        temps.append(handle.name)
        paths.append(Path(handle.name).resolve())
      else:
        paths.append(Path(item).resolve())

    listing = "\n".join(f"- {p}" for p in paths)
    if reference is not None:
      roles = "The first path is the REFERENCE; every path after it is OURS."
    else:
      roles = "Every path is OURS."
    prompt += (
      f"\n\nRead each of these image files in order with the Read tool, then judge.\n"
      f"{roles}\n{listing}\n\n"
      "Output ONLY the JSON object. No preamble, no commentary, no code fence."
    )

    with _tempfile.TemporaryDirectory(prefix="critique-claude-") as workdir:
      # Same locked-down session as read-only agents: Read is the only tool, and
      # it only reaches the scratch cwd and the images' directories.
      cmd = _cli.claude_command(
        judge["model"],
        add_dirs=list(dict.fromkeys(str(p.parent) for p in paths)),
        allowed_tools=("Read",),
        binary=binary,
      )
      completed = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=workdir,
        env=_cli.claude_env(),
      )
  except (OSError, subprocess.TimeoutExpired):
    return {"ok": False, "reason": "cli-error"}
  finally:
    for name in temps:
      Path(name).unlink(missing_ok=True)

  if completed.returncode != 0:
    return {"ok": False, "reason": "cli-error"}
  try:
    parsed = parse_judge_response(
      completed.stdout, recurring_check_units=recurring_check_units
    )
  except (TypeError, ValueError, json.JSONDecodeError):
    return {"ok": False, "reason": "parse-failure"}
  return {"ok": True, **parsed}


def _call_judge(
  client: httpx.Client,
  judge: dict[str, Any],
  messages: list[dict[str, Any]],
  api_key: str,
  recurring_check_units: list[str],
) -> dict[str, Any]:
  payload = {
    "model": judge["model"],
    "messages": messages,
    "temperature": 0,
    # response_format is best-effort: providers that ignore it (e.g. anthropic)
    # fall back to the fenced/embedded JSON extraction in parse_judge_response.
    # require_parameters would hard-400 those providers (found live: anthropic
    # 400 vs gemini 200).
    "response_format": _OPENROUTER_RESPONSE_FORMAT,
    # Ask OpenRouter to report the call's cost so the ledger row is metered.
    "usage": {"include": True},
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
    body = response.json()
  except ValueError:
    return {"ok": False, "reason": "parse-failure"}
  _record_openrouter_cost(judge["model"], body)
  try:
    content = _extract_openrouter_text(body)
    parsed = parse_judge_response(content, recurring_check_units=recurring_check_units)
  except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    return {"ok": False, "reason": "parse-failure"}

  return {"ok": True, **parsed}


def _record_openrouter_cost(model: str, body: Any) -> None:
  """Meter one billed OpenRouter call; the verdict may still fail to parse."""
  if not isinstance(body, dict):
    return
  usage = body.get("usage")
  usage = usage if isinstance(usage, dict) else {}
  cost = usage.get("cost")
  _costs.record(
    source="openrouter",
    model=model,
    usage=usage,
    usd=cost if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None,
    request_id=body.get("id"),
  )


def parse_judge_response(
  text: str,
  *,
  recurring_check_units: list[str] | None = None,
) -> dict[str, Any]:
  """Parse one judge response into a clamped score and normalized defects.

  The verdict is the last JSON object in ``text`` that fits the response schema:
  a finite numeric ``score`` (a number, or a string holding one), a ``defects``
  list and ``recurring_checks``, which must be a list when present. The whole
  text is tried first, then fenced ```json blocks, then every balanced top-level
  object, so narration with braces, a draft before the final answer or a
  trailing note cannot displace the verdict. A verdict that omits
  ``recurring_checks`` still counts; its checks are recorded as skipped.

  Raises:
    ValueError: When no object fits the schema. There is no prose fallback;
      the panel records the judge as ``parse-failure``.
  """
  if not isinstance(text, str):
    raise ValueError("judge response is not text")
  check_units = _normalize_recurring_check_units([b""], recurring_check_units)
  obj = _extract_verdict(text)
  if obj is None:
    raise ValueError("judge response has no JSON verdict with a numeric score and defects list")

  score = _clamp_score(_verdict_score(obj))
  if score is None:
    raise ValueError("judge response missing numeric score")

  defects = [_normalize_defect(d) for d in _verdict_defects(obj)]
  defects.sort(key=lambda d: SEVERITY_RANK[d["severity"]], reverse=True)
  recurring_checks = _normalize_recurring_checks(obj.get("recurring_checks"), check_units)
  return {"score": score, "defects": defects, "recurring_checks": recurring_checks}


_FENCED_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def _extract_verdict(text: str) -> dict[str, Any] | None:
  try:
    whole = json.loads(text.strip())
  except json.JSONDecodeError:
    whole = None
  if _is_verdict(whole):
    return whole
  for candidates in (_fenced_objects(text), _balanced_objects(text)):
    verdicts = [obj for obj in candidates if _is_verdict(obj)]
    if verdicts:
      return verdicts[-1]
  return None


def _fenced_objects(text: str) -> list[Any]:
  objects = []
  for block in _FENCED_BLOCK.findall(text):
    try:
      objects.append(json.loads(block))
    except json.JSONDecodeError:
      continue
  return objects


def _balanced_objects(text: str) -> list[Any]:
  """Every top-level JSON object in ``text``, skipping spans that do not decode."""
  decoder = json.JSONDecoder()
  objects = []
  index = text.find("{")
  while index != -1:
    try:
      obj, end = decoder.raw_decode(text, index)
    except json.JSONDecodeError:
      index = text.find("{", index + 1)
      continue
    objects.append(obj)
    index = text.find("{", end)
  return objects


def _is_verdict(obj: Any) -> bool:
  if not isinstance(obj, dict):
    return False
  score = _verdict_score(obj)
  return (
    not isinstance(score, bool)
    and _clamp_score(score) is not None
    and isinstance(_verdict_defects(obj), list)
    and isinstance(obj.get("recurring_checks", []), list)
  )


def _verdict_score(obj: dict[str, Any]) -> Any:
  # "fidelity"/"findings" are the legacy field names.
  return obj.get("score", obj.get("fidelity"))


def _verdict_defects(obj: dict[str, Any]) -> Any:
  return obj.get("defects", obj.get("findings"))


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
  severity = _SEVERITY_ALIASES.get(str(raw.get("severity", "minor")).strip().lower(), "minor")

  return {
    "key": key,
    "region": _clip(raw.get("region") or raw.get("location") or raw.get("area") or "unspecified"),
    "severity": severity,
    "defect": _clip(defect_text),
    "direction": _clip(direction or ""),
  }


def _normalize_recurring_checks(raw: Any, recurring_check_units: list[str]) -> list[dict[str, Any]]:
  """One judge's checks: one entry per check id for each subject, in subject order."""
  if not isinstance(raw, list):
    return _skipped_recurring_checks(recurring_check_units, "model omitted recurring_checks")

  checks = []
  for item in raw:
    check = _normalize_recurring_check(item)
    if check is not None:
      checks.append(check)
  if not checks:
    return _skipped_recurring_checks(recurring_check_units, "model returned no valid checks")
  return _assign_recurring_check_subjects(checks, recurring_check_units)


def _normalize_recurring_check(raw: Any) -> dict[str, Any] | None:
  if not isinstance(raw, dict):
    return None

  check_id = _normalize_recurring_check_id(raw.get("id"))
  if check_id is None:
    return None

  pass_value = _coerce_check_pass(raw.get("pass"))
  evidence = _clip(raw.get("evidence") or raw.get("location") or raw.get("area") or "")
  if pass_value is None and not evidence.lower().startswith("skipped:"):
    evidence = _clip(f"skipped: {evidence or 'pass value missing'}")

  return {
    "id": check_id,
    # The subject as the model named it; replaced by the matched subject label.
    "subject": raw.get("subject", raw.get("unit")),
    "pass": pass_value,
    "evidence": evidence,
  }


def _normalize_recurring_check_id(value: Any) -> str | None:
  if not isinstance(value, str):
    return None
  compact = re.sub(r"[\s_]+", "-", value.strip().lower())
  return compact if compact in _RECURRING_CHECK_ID_SET else None


def _coerce_check_pass(value: Any) -> bool | None:
  if isinstance(value, bool):
    return value
  if isinstance(value, str):
    normalized = value.strip().lower()
    if normalized in {"pass", "passed", "true", "yes"}:
      return True
    if normalized in {"fail", "failed", "false", "no"}:
      return False
  return None


def _assign_recurring_check_subjects(
  checks: list[dict[str, Any]],
  recurring_check_units: list[str],
) -> list[dict[str, Any]]:
  """Place each check on the subject it names; unnamed checks fill free subjects in order.

  A check whose subject matches no listed subject counts as unnamed. Checks
  beyond the listed subjects become extra subjects ("subject 3", ...). Every
  (subject, check id) the model skipped gets a skipped entry.
  """
  placed: dict[tuple[str, int], dict[str, Any]] = {}
  unnamed: list[dict[str, Any]] = []
  for check in checks:
    index = _subject_index(check["subject"], recurring_check_units)
    if index is None:
      unnamed.append(check)
      continue
    slot = (check["id"], index)
    # A subject named twice keeps the first entry unless a later one fails it.
    if slot not in placed or check["pass"] is False:
      placed[slot] = check
  for check in unnamed:
    index = 0
    while (check["id"], index) in placed:
      index += 1
    placed[(check["id"], index)] = check

  subject_count = max([len(recurring_check_units), *(index + 1 for _id, index in placed)])
  labels = [
    *recurring_check_units,
    *(f"subject {n}" for n in range(len(recurring_check_units) + 1, subject_count + 1)),
  ]
  return [
    {**placed[(check_id, index)], "subject": label}
    if (check_id, index) in placed
    else _skipped_recurring_check(check_id, label, "model omitted check")
    for index, label in enumerate(labels)
    for check_id in RECURRING_CHECK_IDS
  ]


def _subject_index(name: Any, recurring_check_units: list[str]) -> int | None:
  """Index of the listed subject ``name`` refers to, or None when it names none or several.

  Matching ignores case and whitespace; failing an exact match, a subject that
  appears in ``name`` as a whole word or phrase counts when it is the only one.
  """
  if not isinstance(name, str) or not name.strip():
    return None
  wanted = _subject_key(name)
  keys = [_subject_key(unit) for unit in recurring_check_units]
  if wanted in keys:
    return keys.index(wanted)
  matches = [
    index
    for index, key in enumerate(keys)
    if re.search(rf"(?<!\w){re.escape(key)}(?!\w)", wanted)
  ]
  return matches[0] if len(matches) == 1 else None


def _subject_key(value: str) -> str:
  return " ".join(value.lower().split())


def _skipped_recurring_checks(
  recurring_check_units: list[str],
  reason: str,
) -> list[dict[str, Any]]:
  return [
    _skipped_recurring_check(check_id, unit, reason)
    for unit in recurring_check_units
    for check_id in RECURRING_CHECK_IDS
  ]


def _skipped_recurring_check(check_id: str, unit: str, reason: str) -> dict[str, Any]:
  return {
    "id": check_id,
    "subject": unit,
    "pass": None,
    "evidence": _clip(f"skipped: {reason} for {unit}"),
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
  recurring_check_units: list[str],
) -> list[dict[str, Any]]:
  content: list[dict[str, Any]] = [
    {"type": "text", "text": _prompt_with_spec(spec, reference, recurring_check_units)}
  ]
  if reference is not None:
    content.append({"type": "text", "text": "IMAGE 1: REFERENCE"})
    content.append(_image_url_part(reference))
  for index, image in enumerate(images, start=1):
    label = f"OURS {index}" if reference is not None else f"IMAGE {index}: OURS"
    content.append({"type": "text", "text": label})
    content.append(_image_url_part(image))
  return [{"role": "user", "content": content}]


def _prompt_with_spec(
  spec: str | None,
  reference: str | Path | None,
  recurring_check_units: list[str],
) -> str:
  recurring_note = "Recurring check subjects:\n" + "\n".join(
    f"- {unit}" for unit in recurring_check_units
  )
  if spec:
    return f"{CRITIQUE_PROMPT}\n\n{recurring_note}\n\nWritten spec:\n{spec}"
  if reference is None:
    return (
      f"{CRITIQUE_PROMPT}\n\n{recurring_note}\n\nNo reference image or written spec was supplied; "
      "judge internal visual quality and report only concrete defects."
    )
  return f"{CRITIQUE_PROMPT}\n\n{recurring_note}"


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
  for raw in source:
    item: dict[str, Any] = {"id": raw} if isinstance(raw, str) else raw
    # A registry id, as a plain string or a partial dict, keeps the registry's
    # model and transport; fields the caller sets override them. Without this,
    # "codex/gpt-5.6-terra" went to OpenRouter as a bare model id.
    merged = {**registry_by_id.get(item.get("id") or "", {}), **item}
    model = merged.get("model") or merged.get("id")
    judge_id = merged.get("id") or model
    if not model or not judge_id:
      raise ValueError(f"invalid judge registry item: {raw!r}")
    normalized_item = {
      "id": str(judge_id),
      "model": str(model),
      # Listing a judge enables it, even one the registry disables by default.
      "enabled": bool(item.get("enabled", True)),
    }
    # Transport selectors must survive normalization or dispatch falls back
    # to the default OpenRouter path.
    for transport_key in ("cli", "effort", "api"):
      if transport_key in merged:
        normalized_item[transport_key] = merged[transport_key]
    normalized.append(normalized_item)
  return normalized


def _normalize_recurring_check_units(
  images: list[str | Path | bytes | bytearray | memoryview],
  recurring_check_units: list[str] | None,
) -> list[str]:
  if recurring_check_units is None:
    return [f"OURS {index}" for index in range(1, len(images) + 1)]

  units = [str(unit).strip() for unit in recurring_check_units if str(unit).strip()]
  return units or [f"OURS {index}" for index in range(1, len(images) + 1)]


def _budget_allows(
  fn: Callable[..., Any] | None,
  judge: dict[str, Any],
  transport: str,
) -> bool:
  if fn is None:
    return True
  context = {"judge": judge["id"], "model": judge["model"], "transport": transport}
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


def _aggregate_recurring_checks(
  results: list[dict[str, Any]],
  recurring_check_units: list[str],
) -> list[dict[str, Any]]:
  checks_by_slot: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
  for result in results:
    for slot_key, check in _slot_recurring_checks(result["recurring_checks"]).items():
      checks_by_slot[slot_key].append(check)

  slot_count = max([len(recurring_check_units), *(slot + 1 for _id, slot in checks_by_slot)])
  aggregated = []
  for slot in range(slot_count):
    unit = (
      recurring_check_units[slot]
      if slot < len(recurring_check_units)
      else f"subject {slot + 1}"
    )
    for check_id in RECURRING_CHECK_IDS:
      aggregated.append(
        _aggregate_recurring_check_slot(check_id, unit, checks_by_slot.get((check_id, slot), []))
      )
  return aggregated


def _slot_recurring_checks(
  checks: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
  """Key one judge's checks by (check id, subject index).

  Parsing leaves one entry per check id for each subject, in subject order, so
  the n-th entry for an id belongs to subject n.
  """
  slots: dict[tuple[str, int], dict[str, Any]] = {}
  counts: Counter[str] = Counter()
  for check in checks:
    slot = counts[check["id"]]
    counts[check["id"]] += 1
    slots[(check["id"], slot)] = check
  return slots


def _failed_recurring_checks(results: list[dict[str, Any]]) -> list[str]:
  """Check ids that a majority of participants failed for the same subject.

  The threshold is ``ceil(n / 2)`` participants, as for defect consensus. A judge
  that omitted or skipped a check casts no failure vote but still counts in n.
  """
  threshold = math.ceil(len(results) / 2)
  fail_votes: Counter[tuple[str, int]] = Counter()
  for result in results:
    for slot_key, check in _slot_recurring_checks(result["recurring_checks"]).items():
      if check["pass"] is False:
        fail_votes[slot_key] += 1
  failed = {check_id for (check_id, _slot), votes in fail_votes.items() if votes >= threshold}
  return [check_id for check_id in RECURRING_CHECK_IDS if check_id in failed]


def _aggregate_recurring_check_slot(
  check_id: str,
  unit: str,
  checks: list[dict[str, Any]],
) -> dict[str, Any]:
  for check in checks:
    if check["pass"] is False:
      return {"id": check_id, "subject": unit, "pass": False, "evidence": check["evidence"]}
  for check in checks:
    if check["pass"] is True:
      return {"id": check_id, "subject": unit, "pass": True, "evidence": check["evidence"]}
  if checks:
    return {"id": check_id, "subject": unit, "pass": None, "evidence": checks[0]["evidence"]}
  return _skipped_recurring_check(check_id, unit, "model omitted check")


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


def _public_recurring_check(check: dict[str, Any]) -> dict[str, Any]:
  return {
    "id": check["id"],
    "subject": check["subject"],
    "pass": check["pass"],
    "evidence": check["evidence"],
  }


def _clip(value: Any) -> str:
  return str(value or "")[:_MAX_TEXT]


def _openrouter_api_key() -> str | None:
  return _api_key("OPENROUTER_API_KEY")


def _anthropic_api_key() -> str | None:
  return _api_key("ANTHROPIC_API_KEY")


def _api_key(name: str) -> str | None:
  # The package .env is the same file llm.py loads; a hardcoded Mac path broke
  # on Ubuntu, and an explicitly blank variable now disables the judge.
  return _env.provider_key(name)
