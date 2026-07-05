__version__ = "0.1.0"

from merceka_core.agent import (
  Agent,
  AgentComplete,
  AgentProfile,
  AgentRawProviderEvent,
  AgentRequest,
  AgentResult,
  AgentTextDelta,
  ProviderFailure,
  RawProviderEvent,
)
from merceka_core.errors import (
  GpuLockTimeout,
  VideoBackendError,
  VideoNotFoundError,
  VideoUploadError,
)
from merceka_core.resources import GPU_LOCK_PATH, gpu_lock

# Heavy names (LLM pulls litellm/ollama; providers pull subprocess plumbing)
# resolve lazily via PEP 562 so `from merceka_core import gpu_lock` stays light.
_LAZY_EXPORTS = {
  "LLM": ("merceka_core.llm", "LLM"),
  "OutputSchema": ("merceka_core.messages", "OutputSchema"),
  "Tool": ("merceka_core.messages", "Tool"),
  "create_message": ("merceka_core.messages", "create_message"),
  "create_message_with_resource": ("merceka_core.messages", "create_message_with_resource"),
  "create_ollama_vision_message": ("merceka_core.messages", "create_ollama_vision_message"),
  "tool_from_callable": ("merceka_core.messages", "tool_from_callable"),
  "list_local_models": ("merceka_core.llm", "list_local_models"),
  "generate_with_search_grounding": ("merceka_core.llm_gemini", "generate_with_search_grounding"),
  "ClaudeCodeAgentProvider": ("merceka_core.agents.claude_code", "ClaudeCodeAgentProvider"),
  "CodexAgentProvider": ("merceka_core.agents.codex", "CodexAgentProvider"),
  "PiAgentProvider": ("merceka_core.agents.pi", "PiAgentProvider"),
}


def __getattr__(name: str):
  try:
    module_name, attr = _LAZY_EXPORTS[name]
  except KeyError:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
  import importlib
  value = getattr(importlib.import_module(module_name), attr)
  globals()[name] = value  # cache so __getattr__ runs once per name
  return value


def __dir__():
  return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
  "LLM",
  "OutputSchema",
  "Tool",
  "create_message",
  "create_message_with_resource",
  "create_ollama_vision_message",
  "tool_from_callable",
  "list_local_models",
  "generate_with_search_grounding",
  "ClaudeCodeAgentProvider",
  "CodexAgentProvider",
  "PiAgentProvider",
  "Agent",
  "AgentComplete",
  "AgentProfile",
  "AgentRawProviderEvent",
  "AgentRequest",
  "AgentResult",
  "AgentTextDelta",
  "ProviderFailure",
  "RawProviderEvent",
  "GPU_LOCK_PATH",
  "GpuLockTimeout",
  "VideoBackendError",
  "VideoNotFoundError",
  "VideoUploadError",
  "gpu_lock",
]
