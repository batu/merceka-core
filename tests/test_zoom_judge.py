import base64
import io
import json
from unittest.mock import patch

import pytest
from PIL import Image

import importlib

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
    return self._payload


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
