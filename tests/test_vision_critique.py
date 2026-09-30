from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import httpx
import pytest

from merceka_core import costs
from merceka_core.vision import critique, openrouter_budget_floor
from merceka_core.vision import critique as exported_critique
from merceka_core.vision.critique import (
  OPENROUTER_CHAT_URL,
  RECURRING_CHECK_IDS,
  parse_judge_response,
)
from merceka_core.vision import critique as run_critique

# The package re-exports the critique *function* under the submodule's name.
critique_module = importlib.import_module("merceka_core.vision.critique")

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _fake_cli_run(outcomes: dict[str, tuple[int, str]]):
  """subprocess.run stand-in keyed on the CLI binary name: (returncode, answer text)."""
  calls = []

  def run(cmd, **kwargs):
    calls.append((list(cmd), kwargs))
    returncode, text = outcomes.get(Path(cmd[0]).name, (1, ""))
    if "--output-last-message" in cmd:
      Path(cmd[cmd.index("--output-last-message") + 1]).write_text(text)
    return subprocess.CompletedProcess(cmd, returncode, text, "")

  run.calls = calls  # type: ignore[attr-defined]
  return run


def _judge(judge_id: str) -> dict:
  return {"id": judge_id, "model": f"model/{judge_id}", "enabled": True}


def _content(score: float, keys: list[str] | None = None, severity: str = "major") -> str:
  return json.dumps(
    {
      "score": score,
      "defects": [
        {
          "key": key,
          "region": "header",
          "severity": severity,
          "defect": f"{key} differs",
          "direction": f"fix {key}",
        }
        for key in (keys or [])
      ],
    }
  )


def _content_with_defects(score: float, defects: list[dict]) -> str:
  return json.dumps({"score": score, "defects": defects})


def _recurring_checks(unit: str = "OURS 1") -> list[dict]:
  return [
    {
      "id": check_id,
      "pass": check_id != "asset-identity",
      "evidence": f"{unit} x={index} y={index}",
    }
    for index, check_id in enumerate(RECURRING_CHECK_IDS, start=1)
  ]


def _content_with_recurring_checks(score: float, checks: list[dict]) -> str:
  return json.dumps({"score": score, "defects": [], "recurring_checks": checks})


def _openrouter_response(content: str, status_code: int = 200) -> httpx.Response:
  return httpx.Response(
    status_code,
    json={"choices": [{"message": {"content": content}}]},
  )


def _client_for(contents: list[str | httpx.Response | BaseException]) -> httpx.Client:
  calls = []

  def handler(request: httpx.Request) -> httpx.Response:
    calls.append(request)
    item = contents[len(calls) - 1]
    if isinstance(item, BaseException):
      raise item
    if isinstance(item, httpx.Response):
      return item
    return _openrouter_response(item)

  transport = httpx.MockTransport(handler)
  client = httpx.Client(transport=transport)
  client.calls = calls  # type: ignore[attr-defined]
  return client


def test_vision_package_exports_critique():
  assert exported_critique is critique
  assert callable(openrouter_budget_floor)


def test_parse_fenced_json_and_legacy_fidelity_findings():
  parsed = parse_judge_response("""```json
{"fidelity": 91, "findings": [{"key": "layout", "severity": "blocker", "description": "nav shifted", "reference": "centered", "ours": "left"}]}
```""")

  assert parsed["score"] == 91
  assert parsed["defects"] == [
    {
      "key": "layout",
      "region": "unspecified",
      "severity": "blocker",
      "defect": "nav shifted",
      "direction": "reference: centered; ours: left",
    }
  ]


def test_parse_rejects_prose_without_a_json_verdict():
  # There is no prose fallback: a regex used to grab the first number after "score".
  with pytest.raises(ValueError):
    parse_judge_response("Fidelity: 88%\n- Major color: primary button is dull")


@pytest.mark.parametrize(
  "text",
  [
    # Review repro r3 D: the rubric's "0" used to become the score.
    "Score (0-100): 42\n- Blocker background: opaque box behind ribbon",
    # Review repro r4 A: truncated JSON after a prose score used to parse as 97.
    'Score: 97. {"score": 97, "defects": [{"key": "layout"',
  ],
)
def test_parse_prose_scores_are_parse_failures(text):
  with pytest.raises(ValueError):
    parse_judge_response(text)


def test_malformed_json_block_is_a_parse_failure():
  with pytest.raises(ValueError):
    parse_judge_response(
      '```json\n{"score": nope}\n```\nFidelity: 77%\n- Minor spacing: button too low'
    )


def _verdict(score: float, defects: list[dict] | None = None, passes: bool = True) -> str:
  return json.dumps(
    {
      "score": score,
      "defects": defects or [],
      "recurring_checks": [
        {"id": check_id, "pass": passes, "evidence": "OURS 1 x=1 y=1"}
        for check_id in RECURRING_CHECK_IDS
      ],
    }
  )


_BLOCKER = {
  "key": "background",
  "region": "banner",
  "severity": "blocker",
  "defect": "opaque box",
  "direction": "make it transparent",
}


def test_parse_ignores_braced_narration_before_the_verdict():
  # Review repro r3 A: this used to parse as 95.
  text = (
    "Leaning in on the banner. Initial impression: score 95 looks plausible, "
    "but let me check {the ribbon}.\n" + _verdict(41, [_BLOCKER])
  )

  parsed = parse_judge_response(text)

  assert parsed["score"] == 41
  assert [d["severity"] for d in parsed["defects"]] == ["blocker"]
  assert {c["pass"] for c in parsed["recurring_checks"]} == {True}


def test_parse_takes_the_final_verdict_after_a_draft():
  # Review repro r3 B: this used to parse as 96, the draft.
  text = f"Draft:\n{_verdict(96)}\nRevised after zooming:\n{_verdict(38, [_BLOCKER])}"

  parsed = parse_judge_response(text)

  assert parsed["score"] == 38
  assert parsed["defects"][0]["defect"] == "opaque box"


def test_parse_takes_the_last_fenced_verdict():
  text = f"```json\n{_verdict(96)}\n```\nOn reflection:\n```json\n{_verdict(38)}\n```"

  assert parse_judge_response(text)["score"] == 38


def test_parse_keeps_recurring_checks_despite_a_trailing_brace_note():
  # Review repro r6 C: every recurring check used to become None.
  checks = [
    {"id": check_id, "pass": check_id != "banner-transparency", "evidence": "OURS 1"}
    for check_id in RECURRING_CHECK_IDS
  ]
  text = (
    json.dumps({"score": 91, "defects": [], "recurring_checks": checks})
    + "\n(evidence coordinates are in {OURS 1} pixel space)"
  )

  parsed = parse_judge_response(text)

  assert parsed["score"] == 91
  assert parsed["recurring_checks"] == checks


@pytest.mark.parametrize(
  "trailer",
  [
    '{"note": "zoomed twice"}',
    '{"score": 99}',
    '{"score": true, "defects": []}',
    '{"score": 99, "defects": [], "recurring_checks": "all pass"}',
  ],
)
def test_parse_skips_objects_that_are_not_verdicts(trailer):
  parsed = parse_judge_response(f"{_verdict(41)}\n{trailer}")

  assert parsed["score"] == 41


def test_parse_accepts_a_numeric_string_score():
  assert parse_judge_response('{"score": "85", "defects": []}')["score"] == 85


def test_prose_judge_answer_is_skipped_as_parse_failure(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for(["Score (0-100): 42\n- Blocker background: opaque box", _content(92)])

  result = run_critique(
    [PNG_BYTES], judges=[_judge("prose"), _judge("good")], quorum=1, client=client
  )

  assert result["skipped"] == [{"judge": "prose", "reason": "parse-failure"}]
  assert result["score"] == 92


def test_parse_clamps_scores():
  assert parse_judge_response('{"score": 150, "defects": []}')["score"] == 100
  assert parse_judge_response('{"score": -12, "defects": []}')["score"] == 0


def test_critique_median_payload_and_reference(monkeypatch, tmp_path):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  reference = tmp_path / "ref.png"
  reference.write_bytes(PNG_BYTES)
  client = _client_for(
    [
      _content(90, ["layout"]),
      _content(100, ["layout"]),
      _content(100, ["color"], severity="minor"),
    ]
  )

  result = run_critique(
    [PNG_BYTES],
    reference=reference,
    spec="match the reference",
    judges=[_judge("j1"), _judge("j2"), _judge("j3")],
    client=client,
  )

  assert result["score"] == 100
  assert result["verdict"] == "pass"
  assert result["consensus"] == ["layout"]
  assert result["participated"] == ["j1", "j2", "j3"]
  assert result["skipped"] == []
  assert result["defects"] == [
    {
      "region": "header",
      "severity": "major",
      "defect": "layout differs",
      "direction": "fix layout",
    },
    {
      "region": "header",
      "severity": "minor",
      "defect": "color differs",
      "direction": "fix color",
    },
  ]
  assert result["per_model"]["j1"]["defects"][0]["key"] == "layout"
  request = client.calls[0]  # type: ignore[attr-defined]
  assert str(request.url) == OPENROUTER_CHAT_URL
  body = json.loads(request.content)
  assert body["temperature"] == 0
  assert body["response_format"]["type"] == "json_schema"
  # provider.require_parameters dropped: anthropic-via-OpenRouter hard-400s strict
  # params (schema numeric bounds); fenced/embedded JSON extraction is the fallback.
  assert "provider" not in body
  parts = body["messages"][0]["content"]
  assert [part["type"] for part in parts].count("image_url") == 2
  assert parts[2]["image_url"]["url"].startswith("data:image/png;base64,")
  schema = body["response_format"]["json_schema"]["schema"]
  assert schema["properties"]["recurring_checks"]["items"]["properties"]["id"]["enum"] == (
    RECURRING_CHECK_IDS
  )
  prompt = parts[0]["text"]
  assert "banner-transparency" in prompt
  assert "Recurring check subjects:" in prompt


def test_recurring_checks_parse_pass_fail_from_model_response(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for([_content_with_recurring_checks(96, _recurring_checks("hud crop"))])

  result = run_critique(
    [PNG_BYTES],
    recurring_check_units=["hud crop"],
    judges=[_judge("judge")],
    client=client,
  )

  assert result["recurring_checks"] == _recurring_checks("hud crop")
  assert result["per_model"]["judge"]["recurring_checks"] == _recurring_checks("hud crop")


def test_missing_recurring_checks_are_recorded_as_skipped(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for([_content(94, [])])

  result = run_critique(
    [PNG_BYTES],
    recurring_check_units=["hud crop"],
    judges=[_judge("judge")],
    client=client,
  )

  assert [check["id"] for check in result["recurring_checks"]] == RECURRING_CHECK_IDS
  assert all(check["pass"] is None for check in result["recurring_checks"])
  assert all("skipped: model omitted recurring_checks" in check["evidence"] for check in result["recurring_checks"])
  assert result["failed_recurring_checks"] == []
  assert result["verdict"] == "pass"


def _checks_failing(failing: set[str], unit: str = "OURS 1") -> list[dict]:
  return [
    {"id": check_id, "pass": check_id not in failing, "evidence": f"{unit} x=1 y=1"}
    for check_id in RECURRING_CHECK_IDS
  ]


def test_unanimous_recurring_check_failure_fails_the_verdict(monkeypatch):
  # Review repro r2 B: 3/3 judges failing every check at score 92 used to pass.
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  failing = _content_with_recurring_checks(92, _checks_failing(set(RECURRING_CHECK_IDS)))
  client = _client_for([failing, failing, failing])

  result = run_critique(
    [PNG_BYTES], judges=[_judge("j1"), _judge("j2"), _judge("j3")], client=client
  )

  assert result["score"] == 92
  assert result["consensus"] == []
  assert result["failed_recurring_checks"] == RECURRING_CHECK_IDS
  assert result["verdict"] == "fail"


def test_majority_recurring_check_failure_fails_the_verdict(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for(
    [
      _content_with_recurring_checks(95, _checks_failing({"banner-transparency"})),
      _content_with_recurring_checks(95, _checks_failing({"banner-transparency"})),
      _content_with_recurring_checks(95, _checks_failing(set())),
    ]
  )

  result = run_critique(
    [PNG_BYTES], judges=[_judge("j1"), _judge("j2"), _judge("j3")], client=client
  )

  assert result["failed_recurring_checks"] == ["banner-transparency"]
  assert result["verdict"] == "fail"


def test_minority_recurring_check_failure_keeps_the_verdict(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for(
    [
      _content_with_recurring_checks(95, _checks_failing({"banner-transparency"})),
      _content_with_recurring_checks(95, _checks_failing(set())),
      # An omitted check is not a failure vote, but the judge still counts.
      _content(95),
    ]
  )

  result = run_critique(
    [PNG_BYTES], judges=[_judge("j1"), _judge("j2"), _judge("j3")], client=client
  )

  assert result["failed_recurring_checks"] == []
  assert result["verdict"] == "pass"
  banner = next(c for c in result["recurring_checks"] if c["id"] == "banner-transparency")
  assert banner["pass"] is False  # the per-subject table still shows any failure


def test_recurring_check_failures_on_different_subjects_are_not_consensus(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  hud_fails = _checks_failing({"banner-transparency"}, "hud") + _checks_failing(set(), "cta")
  cta_fails = _checks_failing(set(), "hud") + _checks_failing({"banner-transparency"}, "cta")
  all_pass = _checks_failing(set(), "hud") + _checks_failing(set(), "cta")
  client = _client_for(
    [
      _content_with_recurring_checks(95, hud_fails),
      _content_with_recurring_checks(95, cta_fails),
      _content_with_recurring_checks(95, all_pass),
    ]
  )

  result = run_critique(
    [PNG_BYTES],
    recurring_check_units=["hud", "cta"],
    judges=[_judge("j1"), _judge("j2"), _judge("j3")],
    client=client,
  )

  assert result["failed_recurring_checks"] == []
  assert result["verdict"] == "pass"


def test_recurring_checks_gate_off_restores_score_and_blocker_verdict(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  failing = _content_with_recurring_checks(92, _checks_failing({"asset-identity"}))
  client = _client_for([failing, failing])

  result = run_critique(
    [PNG_BYTES],
    judges=[_judge("j1"), _judge("j2")],
    recurring_checks_gate=False,
    client=client,
  )

  assert result["failed_recurring_checks"] == ["asset-identity"]
  assert result["verdict"] == "pass"


def test_openrouter_list_content_is_parsed(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  response = httpx.Response(
    200,
    json={
      "choices": [
        {
          "message": {
            "content": [
              {"type": "text", "text": _content(93, ["typography"])},
            ]
          }
        }
      ]
    },
  )
  client = _client_for([response])

  result = run_critique([PNG_BYTES], judges=[_judge("judge")], client=client)

  assert result["score"] == 93
  assert result["participated"] == ["judge"]
  assert result["consensus"] == ["typography"]


def test_aggregate_defects_preserves_same_key_different_regions(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for(
    [
      _content_with_defects(
        95,
        [
          {
            "key": "layout",
            "region": "image 1 header",
            "severity": "major",
            "defect": "header shifted",
            "direction": "align header",
          },
          {
            "key": "layout",
            "region": "image 2 footer",
            "severity": "minor",
            "defect": "footer shifted",
            "direction": "align footer",
          },
        ],
      )
    ]
  )

  result = run_critique([PNG_BYTES, PNG_BYTES], judges=[_judge("judge")], client=client)

  assert result["defects"] == [
    {
      "region": "image 1 header",
      "severity": "major",
      "defect": "header shifted",
      "direction": "align header",
    },
    {
      "region": "image 2 footer",
      "severity": "minor",
      "defect": "footer shifted",
      "direction": "align footer",
    },
  ]


def test_verdict_fails_below_default_floor(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for([_content(84, [])])

  result = run_critique([PNG_BYTES], judges=[_judge("judge")], client=client)

  assert result["score"] == 84
  assert result["verdict"] == "fail"


def test_verdict_fails_on_consensus_blocker(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for(
    [
      _content(96, ["layout"], severity="blocker"),
      _content(97, ["layout"], severity="blocker"),
      _content(98, []),
    ]
  )

  result = run_critique(
    [PNG_BYTES],
    judges=[_judge("j1"), _judge("j2"), _judge("j3")],
    client=client,
  )

  assert result["score"] == 97
  assert result["consensus"] == ["layout"]
  assert result["verdict"] == "fail"


@pytest.mark.parametrize(
  ("judge_count", "flagged_count", "expected"),
  [
    (1, 1, ["layout"]),
    (3, 2, ["layout"]),
    (7, 4, ["layout"]),
  ],
)
def test_consensus_thresholds_ignore_skipped(monkeypatch, judge_count, flagged_count, expected):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  judges = [_judge(f"j{i}") for i in range(judge_count + 1)]
  contents = []
  for index in range(judge_count):
    keys = ["layout"] if index < flagged_count else ["color"]
    contents.append(_content(90, keys))
  contents.append(httpx.Response(429, text="rate limited"))
  client = _client_for(contents)

  result = run_critique([PNG_BYTES], judges=judges, client=client)

  assert result["consensus"] == expected
  assert len(result["participated"]) == judge_count
  assert result["skipped"] == [{"judge": f"j{judge_count}", "reason": "HTTP 429"}]


@pytest.mark.parametrize("status", [401, 402, 403, 404, 429])
def test_http_skip_classes_are_recorded(monkeypatch, status):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for([httpx.Response(status, text="skip me"), _content(90)])

  result = run_critique([PNG_BYTES], judges=[_judge("bad"), _judge("good")], client=client)

  assert result["participated"] == ["good"]
  assert result["skipped"] == [{"judge": "bad", "reason": f"HTTP {status}"}]
  assert result["per_model"]["bad"]["reason"] == f"HTTP {status}"


def test_timeout_skip_is_recorded(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  request = httpx.Request("POST", OPENROUTER_CHAT_URL)
  client = _client_for([httpx.ReadTimeout("slow", request=request), _content(91)])

  result = run_critique([PNG_BYTES], judges=[_judge("slow"), _judge("good")], client=client)

  assert result["skipped"] == [{"judge": "slow", "reason": "timeout"}]
  assert result["participated"] == ["good"]


def test_parse_failure_skip_is_recorded(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for(["not parseable", _content(92)])

  result = run_critique([PNG_BYTES], judges=[_judge("bad"), _judge("good")], client=client)

  assert result["skipped"] == [{"judge": "bad", "reason": "parse-failure"}]
  assert result["participated"] == ["good"]


def test_empty_choices_skip_parse_failure(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for([httpx.Response(200, json={"choices": []}), _content(92)])

  result = run_critique([PNG_BYTES], judges=[_judge("bad"), _judge("good")], client=client)

  assert result["skipped"] == [{"judge": "bad", "reason": "parse-failure"}]
  assert result["participated"] == ["good"]


def test_budget_halts_remaining_judges_mid_panel(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  decisions = iter([True, False])
  seen = []

  def budget_check(context):
    seen.append(context["judge"])
    return next(decisions)

  client = _client_for([_content(94)])

  result = run_critique(
    [PNG_BYTES],
    judges=[_judge("first"), _judge("second"), _judge("third")],
    budget_check=budget_check,
    quorum=1,
    client=client,
  )

  assert seen == ["first", "second"]
  assert result["participated"] == ["first"]
  assert result["degraded"] is True
  assert result["skipped"] == [
    {"judge": "second", "reason": "budget"},
    {"judge": "third", "reason": "budget"},
  ]
  assert len(client.calls) == 1  # type: ignore[attr-defined]


def test_zero_arg_budget_false_skips_without_chat_call(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for([])

  with pytest.raises(RuntimeError, match="0 participating judges"):
    run_critique(
      [PNG_BYTES],
      judges=[_judge("first"), _judge("second")],
      budget_check=lambda: False,
      client=client,
    )

  assert client.calls == []  # type: ignore[attr-defined]


def test_zero_participants_raise_clear_error(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for([httpx.Response(401, text="bad key"), "not parseable"])

  with pytest.raises(RuntimeError, match="0 participating judges"):
    run_critique([PNG_BYTES], judges=[_judge("auth"), _judge("junk")], client=client)


def test_default_roster_raises_when_most_judges_fail(monkeypatch):
  # Review repro r2 A / r8: expired CLI logins used to leave gemini deciding alone.
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  monkeypatch.setattr(
    critique_module.subprocess, "run", _fake_cli_run({"codex": (1, ""), "claude": (1, "")})
  )
  client = _client_for([_content(86)])

  with pytest.raises(RuntimeError, match="1 participating judges, below the quorum of 2") as exc:
    run_critique([PNG_BYTES], client=client)

  message = str(exc.value)
  assert "codex/gpt-5.6-terra: cli-error" in message
  assert "anthropic/claude-fable-5: cli-error" in message
  assert "anthropic/claude-opus-5: cli-error" in message


def test_quorum_failure_names_skipped_judges_and_reasons(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for([_content(90), httpx.Response(429, text="slow down"), "not parseable"])

  with pytest.raises(RuntimeError, match="below the quorum of 2") as exc:
    run_critique([PNG_BYTES], judges=[_judge("ok"), _judge("rate"), _judge("junk")], client=client)

  assert "rate: HTTP 429" in str(exc.value)
  assert "junk: parse-failure" in str(exc.value)


def test_degraded_panel_reports_participants_and_roster_size(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for([_content(90), httpx.Response(500, text="boom"), _content(92)])

  result = run_critique(
    [PNG_BYTES], judges=[_judge("j1"), _judge("j2"), _judge("j3")], client=client
  )

  assert result["degraded"] is True
  assert result["participants"] == 2
  assert result["roster_size"] == 3
  assert result["skipped"] == [{"judge": "j2", "reason": "HTTP 500"}]


def test_full_panel_is_not_degraded(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for([_content(90), _content(92)])

  result = run_critique([PNG_BYTES], judges=[_judge("j1"), _judge("j2")], client=client)

  assert result["degraded"] is False
  assert result["participants"] == 2
  assert result["roster_size"] == 2


def test_quorum_one_accepts_a_single_surviving_judge(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for([_content(90), httpx.Response(500, text="boom"), "not parseable"])

  result = run_critique(
    [PNG_BYTES], judges=[_judge("j1"), _judge("j2"), _judge("j3")], quorum=1, client=client
  )

  assert result["participated"] == ["j1"]
  assert result["degraded"] is True
  assert result["participants"] == 1
  assert result["roster_size"] == 3


def test_disabled_judges_do_not_count_toward_the_roster(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for([_content(90)])

  result = run_critique(
    [PNG_BYTES], judges=[_judge("j1"), dict(_judge("off"), enabled=False)], client=client
  )

  assert result["skipped"] == [{"judge": "off", "reason": "disabled"}]
  assert result["degraded"] is False
  assert result["roster_size"] == 1
  assert result["participants"] == 1


def test_unreachable_quorum_raises_before_any_judge_call(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for([])

  with pytest.raises(ValueError, match="quorum 3 exceeds the 2 enabled judges"):
    run_critique([PNG_BYTES], judges=[_judge("j1"), _judge("j2")], quorum=3, client=client)

  assert client.calls == []  # type: ignore[attr-defined]


def test_explicit_string_roster_keeps_registry_transports(monkeypatch):
  # Review repro r1: plain ids lost cli/effort/api and went to OpenRouter as bare models.
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  fake_run = _fake_cli_run({"codex": (0, _content(40)), "claude": (0, _content(40))})
  monkeypatch.setattr(critique_module.subprocess, "run", fake_run)
  client = _client_for([_content(92)])

  result = run_critique(
    [PNG_BYTES],
    judges=[
      "codex/gpt-5.6-terra",
      "anthropic/claude-fable-5",
      "anthropic/claude-opus-5",
      "google/gemini-3.6-flash",
    ],
    client=client,
  )

  assert result["participants"] == 4
  assert result["score"] == 40
  sent = [json.loads(request.content)["model"] for request in client.calls]  # type: ignore[attr-defined]
  assert sent == ["google/gemini-3.6-flash"]
  binaries = [Path(cmd[0]).name for cmd, _kwargs in fake_run.calls]  # type: ignore[attr-defined]
  assert binaries == ["codex", "claude", "claude"]
  codex_cmd = fake_run.calls[0][0]  # type: ignore[attr-defined]
  assert any("model_reasoning_effort" in arg and "max" in arg for arg in codex_cmd)


def test_explicit_roster_normalizes_like_the_default_roster():
  from_registry = critique_module._normalize_judges(None)
  explicit = critique_module._normalize_judges([judge["id"] for judge in from_registry])

  assert explicit == from_registry


@pytest.mark.parametrize(
  "item",
  [
    "anthropic/claude-fable-5-zoom",
    {"id": "anthropic/claude-fable-5-zoom", "enabled": True},
  ],
)
def test_explicit_roster_enables_registry_judges_disabled_by_default(item):
  (judge,) = critique_module._normalize_judges([item])

  assert judge == {
    "id": "anthropic/claude-fable-5-zoom",
    "model": "claude-fable-5",
    "api": "anthropic-zoom",
    "enabled": True,
  }


def test_partial_dict_overrides_registry_fields():
  (judge,) = critique_module._normalize_judges([{"id": "codex/gpt-5.6-terra", "effort": "low"}])

  assert judge == {
    "id": "codex/gpt-5.6-terra",
    "model": "gpt-5.6-terra",
    "cli": "codex",
    "effort": "low",
    "enabled": True,
  }


def test_unknown_string_judge_goes_to_openrouter_by_id():
  (judge,) = critique_module._normalize_judges(["vendor/new-model"])

  assert judge == {"id": "vendor/new-model", "model": "vendor/new-model", "enabled": True}


def _ledger_rows() -> list[dict]:
  path = costs.ledger_path()
  if not path.exists():
    return []
  return [json.loads(line) for line in path.read_text().splitlines()]


def _metered_response(content: str, generation_id: str, cost: float) -> httpx.Response:
  return httpx.Response(
    200,
    json={
      "id": generation_id,
      "choices": [{"message": {"content": content}}],
      "usage": {"prompt_tokens": 1200, "completion_tokens": 300, "cost": cost},
    },
  )


def test_openrouter_judge_calls_are_metered_and_cli_judges_are_not(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  monkeypatch.setattr(critique_module.subprocess, "run", _fake_cli_run({"codex": (0, _content(90))}))
  client = _client_for(
    [
      _metered_response(_content(92), "gen-1", 0.0123),
      # Billed even though the verdict does not parse.
      _metered_response("not parseable", "gen-2", 0.004),
      httpx.Response(429, text="rate limited"),
    ]
  )

  result = run_critique(
    [PNG_BYTES],
    judges=["codex/gpt-5.6-terra", _judge("j1"), _judge("junk"), _judge("limited")],
    quorum=1,
    client=client,
  )

  assert result["participated"] == ["codex/gpt-5.6-terra", "j1"]
  body = json.loads(client.calls[0].content)  # type: ignore[attr-defined]
  assert body["usage"] == {"include": True}
  rows = [{k: v for k, v in row.items() if k != "ts"} for row in _ledger_rows()]
  usage = {"prompt_tokens": 1200, "completion_tokens": 300}
  assert rows == [
    {
      "source": "openrouter",
      "model": "model/j1",
      "usage": {**usage, "cost": 0.0123},
      "usd": 0.0123,
      "usd_source": "provider",
      "request_id": "gen-1",
    },
    {
      "source": "openrouter",
      "model": "model/junk",
      "usage": {**usage, "cost": 0.004},
      "usd": 0.004,
      "usd_source": "provider",
      "request_id": "gen-2",
    },
  ]


def test_openrouter_response_without_usage_is_recorded_unpriced(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = _client_for([_content(92)])

  run_critique([PNG_BYTES], judges=[_judge("j1")], client=client)

  (row,) = _ledger_rows()
  assert row["source"] == "openrouter"
  assert row["model"] == "model/j1"
  assert row["usage"] == {}
  assert row["usd"] is None
  assert "request_id" not in row


def test_budget_floor_uses_openrouter_credits(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

  def handler(request: httpx.Request) -> httpx.Response:
    assert str(request.url) == "https://openrouter.ai/api/v1/credits"
    return httpx.Response(200, json={"data": {"total_credits": 10.0, "total_usage": 4.75}})

  client = httpx.Client(transport=httpx.MockTransport(handler))
  check = openrouter_budget_floor(5.0, client=client)

  assert check() is True


def test_budget_floor_returns_false_below_floor(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = httpx.Client(
    transport=httpx.MockTransport(
      lambda _request: httpx.Response(
        200,
        json={"data": {"total_credits": 10.0, "total_usage": 6.0}},
      )
    )
  )

  assert openrouter_budget_floor(5.0, client=client)() is False


@pytest.mark.parametrize(
  "response",
  [
    httpx.Response(403, text="forbidden"),
    httpx.Response(200, json=[]),
    httpx.Response(200, json={"data": []}),
    httpx.Response(200, content=b"not-json"),
    httpx.Response(200, json={"data": {"total_credits": "bad", "total_usage": 1}}),
  ],
)
def test_budget_floor_returns_false_on_unusable_credit_response(monkeypatch, response):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  client = httpx.Client(transport=httpx.MockTransport(lambda _request: response))

  assert openrouter_budget_floor(5.0, client=client)() is False
