from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from merceka_core.agent import (
  Agent,
  AgentComplete,
  AgentProfile,
  AgentRawProviderEvent,
  AgentRequest,
  AgentResult,
  AgentTextDelta,
  RawProviderEvent,
)


@dataclass
class FakeProvider:
  result: AgentResult
  stream_events: list[AgentTextDelta | AgentRawProviderEvent | AgentComplete]
  received_request: AgentRequest | None = None

  async def run(self, request: AgentRequest) -> AgentResult:
    self.received_request = request
    return self.result

  async def stream(
    self, request: AgentRequest
  ) -> AsyncIterator[AgentTextDelta | AgentRawProviderEvent | AgentComplete]:
    self.received_request = request
    for event in self.stream_events:
      yield event


@pytest.mark.asyncio
async def test_agent_run_passes_read_only_request_and_retains_raw_events(tmp_path: Path):
  raw_event = RawProviderEvent(
    provider="fake",
    event_type="result",
    payload={"native": {"answer": "hello"}},
  )
  provider = FakeProvider(
    result=AgentResult(text="hello", raw_events=(raw_event,)),
    stream_events=[],
  )
  request = AgentRequest(
    message="What happened?",
    system_prompt="Read before answering.",
    roots=(tmp_path,),
    profile=AgentProfile.READ_ONLY,
  )

  result = await Agent(provider).run(request)

  assert result.text == "hello"
  assert result.raw_events == (raw_event,)
  assert provider.received_request == request


@pytest.mark.asyncio
async def test_agent_stream_yields_text_raw_events_and_completion(tmp_path: Path):
  raw_event = RawProviderEvent(
    provider="fake",
    event_type="tool_use",
    payload={"tool": "read", "path": "chapter.md"},
  )
  result = AgentResult(text="hello world", raw_events=(raw_event,))
  provider = FakeProvider(
    result=result,
    stream_events=[
      AgentTextDelta(content="hello "),
      AgentRawProviderEvent(raw_event=raw_event),
      AgentTextDelta(content="world"),
      AgentComplete(result=result),
    ],
  )
  request = AgentRequest(message="Question", system_prompt="Prompt", roots=(tmp_path,))

  events = [event async for event in Agent(provider).stream(request)]

  assert events == provider.stream_events


def test_agent_request_rejects_unsupported_profile(tmp_path: Path):
  with pytest.raises(ValueError, match="Unsupported agent profile"):
    AgentRequest(
      message="Question",
      system_prompt="Prompt",
      roots=(tmp_path,),
      profile="write_enabled",  # type: ignore[arg-type]
    )


def test_agent_request_requires_declared_roots_for_file_grounding():
  with pytest.raises(ValueError, match="at least one declared root"):
    AgentRequest(message="Question", system_prompt="Prompt", roots=())


def test_agent_request_rejects_missing_roots(tmp_path: Path):
  missing_root = tmp_path / "missing"

  with pytest.raises(ValueError, match="Declared agent root does not exist"):
    AgentRequest(message="Question", system_prompt="Prompt", roots=(missing_root,))


def test_raw_provider_event_keeps_payload_opaque():
  payload = {"provider_specific": {"nested": [1, 2, 3]}}
  event = RawProviderEvent(provider="fake", event_type="raw", payload=payload)

  assert event.payload is payload
