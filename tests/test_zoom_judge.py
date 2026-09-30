import base64
import functools
import io
import json
from unittest.mock import patch

import httpx
import pytest
from PIL import Image

import importlib

from merceka_core import costs

# The vision package re-exports the critique *function* under the same name as
# the submodule, so attribute-style imports resolve to the function.
critique_module = importlib.import_module("merceka_core.vision.critique")
from merceka_core.vision import zoom_judge
from merceka_core.vision.critique import critique as run_critique


def _png_bytes(size=(64, 64), color="red") -> bytes:
  buf = io.BytesIO()
  Image.new("RGB", size, color).save(buf, format="PNG")
  return buf.getvalue()


_VERDICT = {
  "score": 88,
  "defects": [],
  "recurring_checks": [],
}


class _FakeResponse:
  status_code = 200

  def __init__(self, payload):
    self._payload = payload

  def json(self):
    if isinstance(self._payload, _NotJSON):
      raise json.JSONDecodeError("Expecting value", "<html>gateway</html>", 0)
    return self._payload


class _NotJSON:
  """A 200 whose body is not JSON, e.g. a gateway error page."""


class _FakeClient:
  """Anthropic Messages API stand-in: one tool_use round, then a verdict."""

  def __init__(self, responses):
    self.responses = list(responses)
    self.requests = []

  def post(self, url, json=None, headers=None):
    self.requests.append({"url": url, "json": json, "headers": headers})
    return _FakeResponse(self.responses.pop(0))


def _tool_use_response(box, image_index=0):
  return {
    "stop_reason": "tool_use",
    "content": [
      {"type": "text", "text": "Leaning in."},
      {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "zoom",
        "input": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3], "image_index": image_index},
      },
    ],
  }


def _final_response():
  return {
    "stop_reason": "end_turn",
    "content": [{"type": "text", "text": json.dumps(_VERDICT)}],
  }


ZOOM_JUDGE = {"id": "anthropic/claude-fable-5-zoom", "model": "claude-fable-5", "api": "anthropic-zoom"}


def test_zoom_judge_runs_tool_loop_and_returns_final_text():
  client = _FakeClient([_tool_use_response((0, 0, 32, 32)), _final_response()])

  result = zoom_judge.call_zoom_judge(
    ZOOM_JUDGE, [_png_bytes()], None, "judge this", api_key="sk-ant-test", client=client
  )

  assert result == {"ok": True, "text": json.dumps(_VERDICT)}
  assert len(client.requests) == 2
  first = client.requests[0]["json"]
  assert first["model"] == "claude-fable-5"
  assert first["tools"][0]["name"] == "zoom"
  assert client.requests[0]["headers"]["x-api-key"] == "sk-ant-test"
  # Second request carries the assistant tool_use turn plus our tool_result
  followup = client.requests[1]["json"]["messages"]
  assert followup[1]["role"] == "assistant"
  tool_result = followup[2]["content"][0]
  assert tool_result["type"] == "tool_result"
  assert tool_result["tool_use_id"] == "toolu_1"
  crop_block = tool_result["content"][1]
  assert crop_block["source"]["media_type"] == "image/jpeg"
  crop = Image.open(io.BytesIO(base64.b64decode(crop_block["source"]["data"])))
  assert crop.width > 32  # magnified, not returned at crop size


def test_zoom_judge_reports_bad_boxes_as_tool_errors_and_continues():
  client = _FakeClient([_tool_use_response((90, 90, 10, 10)), _final_response()])

  result = zoom_judge.call_zoom_judge(
    ZOOM_JUDGE, [_png_bytes()], None, "judge this", api_key="sk-ant-test", client=client
  )

  assert result["ok"] is True
  tool_result = client.requests[1]["json"]["messages"][2]["content"][0]
  assert tool_result["is_error"] is True


def test_zoom_judge_labels_reference_before_ours():
  client = _FakeClient([_final_response()])

  zoom_judge.call_zoom_judge(
    ZOOM_JUDGE, [_png_bytes()], _png_bytes(color="blue"), "judge this",
    api_key="sk-ant-test", client=client,
  )

  texts = [
    block["text"]
    for block in client.requests[0]["json"]["messages"][0]["content"]
    if block["type"] == "text"
  ]
  assert any("REFERENCE" in text for text in texts)
  assert any("OURS" in text for text in texts)


def test_zoom_judge_gives_up_after_max_rounds():
  client = _FakeClient([_tool_use_response((0, 0, 8, 8))] * zoom_judge._MAX_ROUNDS)

  result = zoom_judge.call_zoom_judge(
    ZOOM_JUDGE, [_png_bytes()], None, "judge this", api_key="sk-ant-test", client=client
  )

  assert result == {"ok": False, "reason": "max-rounds"}


def test_critique_skips_zoom_judge_without_anthropic_key(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

  with pytest.raises(RuntimeError, match="0 participating judges"):
    run_critique([_png_bytes()], judges=[dict(ZOOM_JUDGE, enabled=True)])


def test_critique_parses_zoom_judge_verdict(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

  with patch.object(
    critique_module._zoom_judge,
    "call_zoom_judge",
    return_value={"ok": True, "text": json.dumps(_VERDICT)},
  ):
    result = run_critique([_png_bytes()], judges=[dict(ZOOM_JUDGE, enabled=True)])

  assert result["score"] == 88
  assert result["participated"] == ["anthropic/claude-fable-5-zoom"]


def _rgba_png_bytes() -> bytes:
  # Transparent canvas with an opaque white square in the middle (review repro r4 G).
  image = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
  image.paste(Image.new("RGBA", (50, 50), (255, 255, 255, 255)), (75, 75))
  buf = io.BytesIO()
  image.save(buf, format="PNG")
  return buf.getvalue()


def _returned_crop(client) -> tuple[str, Image.Image]:
  tool_result = client.requests[1]["json"]["messages"][2]["content"][0]
  label, block = tool_result["content"]
  assert block["source"]["media_type"] == "image/jpeg"
  return label["text"], Image.open(io.BytesIO(base64.b64decode(block["source"]["data"])))


def test_zoom_crop_draws_transparency_on_a_checkerboard_not_black():
  client = _FakeClient([_tool_use_response((60, 60, 140, 140)), _final_response()])

  result = zoom_judge.call_zoom_judge(
    ZOOM_JUDGE, [_rgba_png_bytes()], None, "judge this", api_key="sk-ant-test", client=client
  )

  assert result["ok"] is True
  label, crop = _returned_crop(client)
  assert "checkerboard" in label
  gray = crop.convert("L")
  # The top-left eighth of the crop is fully transparent in the source.
  margin = gray.crop((0, 0, gray.width // 8, gray.height // 8))
  low, high = margin.getextrema()
  assert low > 50  # convert("RGB") used to paint this black (0)
  assert high - low > 30  # a pattern, not a flat fill
  assert gray.getpixel((gray.width // 2, gray.height // 2)) > 240  # opaque white stays white


def test_opaque_zoom_crop_has_no_checkerboard_note():
  client = _FakeClient([_tool_use_response((0, 0, 32, 32)), _final_response()])

  zoom_judge.call_zoom_judge(
    ZOOM_JUDGE, [_png_bytes()], None, "judge this", api_key="sk-ant-test", client=client
  )

  label, crop = _returned_crop(client)
  assert "checkerboard" not in label
  red, green, blue = crop.convert("RGB").getpixel((crop.width // 2, crop.height // 2))
  assert red > 240 and green < 20 and blue < 20


@pytest.mark.parametrize(
  ("stop_reason", "reason"), [("max_tokens", "max-tokens"), ("refusal", "refusal")]
)
def test_zoom_judge_treats_truncated_or_refused_turns_as_failures(stop_reason, reason):
  # Review repro r4 A: truncated JSON after "Score: 97." used to count as a verdict.
  client = _FakeClient(
    [
      {
        "stop_reason": stop_reason,
        "content": [{"type": "text", "text": 'Score: 97. {"score": 97, "defects": [{"key"'}],
      }
    ]
  )

  result = zoom_judge.call_zoom_judge(
    ZOOM_JUDGE, [_png_bytes()], None, "judge this", api_key="sk-ant-test", client=client
  )

  assert result == {"ok": False, "reason": reason}


@pytest.mark.parametrize(
  "body",
  [
    _NotJSON(),
    [],
    {"type": "error"},
    {"content": None, "stop_reason": "end_turn"},
    {"content": ["x"], "stop_reason": "end_turn"},
    {"content": [{"type": "tool_use", "id": "t1", "input": "zoom"}], "stop_reason": "tool_use"},
    {"content": [{"type": "text", "text": "zooming"}], "stop_reason": "tool_use"},
  ],
)
def test_zoom_judge_reports_malformed_responses_instead_of_raising(body):
  client = _FakeClient([body])

  result = zoom_judge.call_zoom_judge(
    ZOOM_JUDGE, [_png_bytes()], None, "judge this", api_key="sk-ant-test", client=client
  )

  assert result == {"ok": False, "reason": "malformed-response"}


@pytest.mark.parametrize(
  "tool_input",
  [
    {"x1": 0, "y1": 0, "x2": 8, "y2": 8, "image_index": "abc"},
    {"x1": 0, "y1": 0, "x2": 8, "y2": 8, "image_index": [0]},
    {"x1": "left", "y1": 0, "x2": 8, "y2": 8},
  ],
)
def test_zoom_judge_returns_bad_tool_arguments_as_tool_errors(tool_input):
  # Review repro r4 C: these raised out of the loop and aborted the whole panel.
  tool_use = {
    "stop_reason": "tool_use",
    "content": [{"type": "tool_use", "id": "toolu_1", "name": "zoom", "input": tool_input}],
  }
  client = _FakeClient([tool_use, _final_response()])

  result = zoom_judge.call_zoom_judge(
    ZOOM_JUDGE, [_png_bytes()], None, "judge this", api_key="sk-ant-test", client=client
  )

  assert result["ok"] is True
  tool_result = client.requests[1]["json"]["messages"][2]["content"][0]
  assert tool_result["is_error"] is True
  assert tool_result["content"][0]["text"].startswith("Error:")


def test_zoom_judge_accepts_cmyk_input():
  # Review repro r4 F: CMYK could not be written as PNG and raised.
  buf = io.BytesIO()
  Image.new("CMYK", (40, 40), (0, 255, 255, 0)).save(buf, format="JPEG")
  client = _FakeClient([_final_response()])

  result = zoom_judge.call_zoom_judge(
    ZOOM_JUDGE, [buf.getvalue()], None, "judge this", api_key="sk-ant-test", client=client
  )

  assert result["ok"] is True


def test_zoom_judge_reports_unreadable_images_instead_of_raising():
  client = _FakeClient([])

  result = zoom_judge.call_zoom_judge(
    ZOOM_JUDGE, [b"not an image"], None, "judge this", api_key="sk-ant-test", client=client
  )

  assert result == {"ok": False, "reason": "image-error"}
  assert client.requests == []


def test_critique_keeps_finished_judges_when_the_zoom_judge_fails(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
  monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
  zoom_client = _FakeClient([_NotJSON()])
  monkeypatch.setattr(
    zoom_judge,
    "call_zoom_judge",
    functools.partial(zoom_judge.call_zoom_judge, client=zoom_client),
  )
  openrouter = httpx.Client(
    transport=httpx.MockTransport(
      lambda _request: httpx.Response(
        200, json={"choices": [{"message": {"content": json.dumps(_VERDICT)}}]}
      )
    )
  )

  result = run_critique(
    [_png_bytes()],
    judges=[{"id": "panel/or", "model": "vendor/or"}, dict(ZOOM_JUDGE, enabled=True)],
    client=openrouter,
  )

  assert result["participated"] == ["panel/or"]
  assert result["skipped"] == [{"judge": ZOOM_JUDGE["id"], "reason": "malformed-response"}]


def _metered(payload: dict, message_id: str, input_tokens: int = 1500) -> dict:
  usage = {"input_tokens": input_tokens, "output_tokens": 120}
  return {**payload, "id": message_id, "usage": usage}


def _zoom_ledger_rows() -> list[dict]:
  path = costs.ledger_path()
  if not path.exists():
    return []
  rows = [json.loads(line) for line in path.read_text().splitlines()]
  return [{k: v for k, v in row.items() if k != "ts"} for row in rows]


def test_every_billed_zoom_round_is_metered():
  client = _FakeClient(
    [
      _metered(_tool_use_response((0, 0, 32, 32)), "msg_1"),
      _metered(_final_response(), "msg_2", input_tokens=4200),
    ]
  )

  result = zoom_judge.call_zoom_judge(
    ZOOM_JUDGE, [_png_bytes()], None, "judge this", api_key="sk-ant-test", client=client
  )

  assert result["ok"] is True
  assert _zoom_ledger_rows() == [
    {
      "source": "anthropic",
      "model": "claude-fable-5",
      "usage": {"input_tokens": 1500, "output_tokens": 120},
      "usd": None,
      "request_id": "msg_1",
    },
    {
      "source": "anthropic",
      "model": "claude-fable-5",
      "usage": {"input_tokens": 4200, "output_tokens": 120},
      "usd": None,
      "request_id": "msg_2",
    },
  ]


def test_truncated_zoom_turn_is_still_metered():
  client = _FakeClient(
    [_metered({"stop_reason": "max_tokens", "content": [{"type": "text", "text": "{"}]}, "msg_9")]
  )

  result = zoom_judge.call_zoom_judge(
    ZOOM_JUDGE, [_png_bytes()], None, "judge this", api_key="sk-ant-test", client=client
  )

  assert result == {"ok": False, "reason": "max-tokens"}
  assert [row["request_id"] for row in _zoom_ledger_rows()] == ["msg_9"]


def test_unparseable_zoom_response_writes_no_ledger_row():
  client = _FakeClient([_NotJSON()])

  zoom_judge.call_zoom_judge(
    ZOOM_JUDGE, [_png_bytes()], None, "judge this", api_key="sk-ant-test", client=client
  )

  assert _zoom_ledger_rows() == []
