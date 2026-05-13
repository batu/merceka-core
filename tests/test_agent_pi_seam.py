from __future__ import annotations

from merceka_core.agent import AgentComplete, AgentResult, AgentTextDelta, RawProviderEvent


def test_pi_style_text_delta_fits_normalized_stream_event():
  fake_pi_event = {
    "type": "response.output_text.delta",
    "delta": "hello",
    "sequence": 3,
  }

  normalized = AgentTextDelta(content=fake_pi_event["delta"])

  assert normalized.type == "text_delta"
  assert normalized.content == "hello"


def test_pi_style_tool_events_remain_opaque_raw_provider_events():
  fake_pi_event = {
    "type": "tool_call",
    "tool": "read",
    "arguments": {"path": "_index.md"},
  }

  raw = RawProviderEvent(
    provider="pi",
    event_type=fake_pi_event["type"],
    payload=fake_pi_event,
  )

  assert raw.provider == "pi"
  assert raw.event_type == "tool_call"
  assert raw.payload is fake_pi_event
  assert not hasattr(raw, "tool")


def test_pi_style_completion_fits_agent_result_without_pi_fields():
  fake_pi_result = {
    "type": "turn.complete",
    "usage": {"input_tokens": 10, "output_tokens": 4},
    "final_text": "done",
  }
  raw = RawProviderEvent(
    provider="pi",
    event_type="turn.complete",
    payload=fake_pi_result,
  )

  completion = AgentComplete(result=AgentResult(text="done", raw_events=(raw,)))

  assert completion.type == "complete"
  assert completion.result.text == "done"
  assert completion.result.raw_events[0].payload is fake_pi_result
