from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol


class AgentProfile(StrEnum):
  READ_ONLY = "read_only"
  WRITE = "write"


class ProviderFailure(RuntimeError):
  """Raised when an agent provider cannot complete a request."""


@dataclass(frozen=True)
class RawProviderEvent:
  provider: str
  event_type: str
  payload: Any
  metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResult:
  text: str
  raw_events: tuple[RawProviderEvent, ...] = ()


@dataclass(frozen=True)
class AgentRequest:
  message: str
  system_prompt: str
  roots: tuple[Path, ...]
  profile: AgentProfile | str = AgentProfile.READ_ONLY

  def __post_init__(self) -> None:
    try:
      profile = AgentProfile(self.profile)
    except ValueError:
      raise ValueError(f"Unsupported agent profile: {self.profile}") from None
    if not self.roots:
      raise ValueError("Agent requests require at least one declared root")

    normalized_roots = tuple(Path(root).resolve() for root in self.roots)
    for root in normalized_roots:
      if not root.exists():
        raise ValueError(f"Declared agent root does not exist: {root}")
      if not root.is_dir():
        raise ValueError(f"Declared agent root is not a directory: {root}")

    object.__setattr__(self, "profile", profile)
    object.__setattr__(self, "roots", normalized_roots)


@dataclass(frozen=True)
class AgentTextDelta:
  content: str
  type: str = "text_delta"


@dataclass(frozen=True)
class AgentRawProviderEvent:
  raw_event: RawProviderEvent
  type: str = "raw_provider_event"


@dataclass(frozen=True)
class AgentComplete:
  result: AgentResult
  type: str = "complete"


AgentStreamEvent = AgentTextDelta | AgentRawProviderEvent | AgentComplete


class AgentProvider(Protocol):
  async def run(self, request: AgentRequest) -> AgentResult:
    ...

  def stream(self, request: AgentRequest) -> AsyncIterator[AgentStreamEvent]:
    ...


@dataclass(frozen=True)
class Agent:
  provider: AgentProvider

  async def run(self, request: AgentRequest) -> AgentResult:
    return await self.provider.run(request)

  def stream(self, request: AgentRequest) -> AsyncIterator[AgentStreamEvent]:
    return self.provider.stream(request)
