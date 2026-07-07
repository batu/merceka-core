from __future__ import annotations

import json

import httpx
import pytest

from merceka_core.vision import critique, openrouter_budget_floor
from merceka_core.vision import critique as exported_critique
from merceka_core.vision.critique import OPENROUTER_CHAT_URL, parse_judge_response
from merceka_core.vision import critique as run_critique


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


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


def test_parse_prose_fallback():
  parsed = parse_judge_response("Fidelity: 88%\n- Major color: primary button is dull")

  assert parsed["score"] == 88
  assert parsed["defects"][0]["key"] == "color"
  assert parsed["defects"][0]["severity"] == "major"


def test_malformed_json_block_falls_back_to_prose():
  parsed = parse_judge_response(
    '```json\n{"score": nope}\n```\nFidelity: 77%\n- Minor spacing: button too low'
  )

  assert parsed["score"] == 77
  assert parsed["defects"][0]["key"] == "spacing"


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
  assert body["provider"] == {"require_parameters": True}
  parts = body["messages"][0]["content"]
  assert [part["type"] for part in parts].count("image_url") == 2
  assert parts[2]["image_url"]["url"].startswith("data:image/png;base64,")


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
    client=client,
  )

  assert seen == ["first", "second"]
  assert result["participated"] == ["first"]
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
