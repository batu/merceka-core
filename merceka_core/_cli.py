"""Shared CLI knowledge for the Claude Code and Codex subprocess providers.

Both layers that shell out to the CLIs — the lightweight text paths in
``llm.py`` and the rooted agent providers in ``agents/`` — build their
commands, environment, and stream parsing here, so flag knowledge cannot
drift between them. The layers keep different *semantics* (plain text calls
grant no tool access; agent requests are rooted and profiled); those
differences are explicit parameters below, not parallel copies.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = [
  "claude_command",
  "claude_env",
  "claude_stream_text_delta",
  "codex_exec_command",
  "is_claude_result_event",
]


def claude_command(
  model: str,
  *,
  system_prompt: str = "",
  add_dirs: list[str] | tuple[str, ...] = (),
  allowed_tools: list[str] | tuple[str, ...] = (),
  stream: bool = False,
  accept_edits: bool = False,
  binary: str = "claude",
) -> list[str]:
  """Build a `claude -p` command. The prompt is passed on stdin by the caller."""
  cmd = [binary, "-p", "--model", model]
  if stream:
    cmd.extend([
      "--output-format", "stream-json",
      "--verbose",
      "--include-partial-messages",
    ])
  if accept_edits:
    cmd.extend(["--permission-mode", "acceptEdits"])
  if system_prompt:
    cmd.extend(["--system-prompt", system_prompt])
  for d in add_dirs:
    cmd.extend(["--add-dir", str(d)])
  if allowed_tools:
    cmd.extend(["--allowedTools", ",".join(allowed_tools)])
  return cmd


def claude_env() -> dict[str, str]:
  """Environment for Claude CLI runs: blank the API key so the CLI uses
  subscription auth instead of accidental API billing."""
  return {**os.environ, "ANTHROPIC_API_KEY": ""}


def codex_exec_command(
  model: str = "",
  *,
  ephemeral: bool = False,
  sandbox: str = "read-only",
  cd: str | None = None,
  add_dirs: list[str] | tuple[str, ...] = (),
  images: list[str] | tuple[str, ...] = (),
  json_output: bool = False,
  reasoning_effort: str | None = None,
  binary: str = "codex",
) -> list[str]:
  """Build a `codex exec` command ending in `-` (prompt on stdin).

  ``model=""`` or ``"default"`` uses the user's configured default model
  (optionally with ``reasoning_effort``); anything else is passed explicitly.
  """
  cmd = [binary, "exec"]
  if ephemeral:
    cmd.append("--ephemeral")
  if model and model != "default":
    cmd.extend(["--model", model])
  elif reasoning_effort:
    cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
  cmd.extend(["--sandbox", sandbox])
  if cd is not None:
    cmd.extend(["--cd", str(cd)])
  cmd.extend(["--skip-git-repo-check", "--color", "never"])
  if json_output:
    cmd.append("--json")
  for d in add_dirs:
    cmd.extend(["--add-dir", str(d)])
  for img in images:
    cmd.extend(["-i", str(img)])
  cmd.append("-")
  return cmd


def claude_stream_text_delta(payload: dict[str, Any]) -> str | None:
  """Extract the text delta from a claude stream-json event, if any."""
  if payload.get("type") != "stream_event":
    return None
  event = payload.get("event")
  if not isinstance(event, dict) or event.get("type") != "content_block_delta":
    return None
  delta = event.get("delta")
  if not isinstance(delta, dict) or delta.get("type") != "text_delta":
    return None
  text = delta.get("text")
  return text if isinstance(text, str) else None


def is_claude_result_event(payload: dict[str, Any]) -> bool:
  """True when the stream-json event marks the end of the response."""
  return payload.get("type") == "result"
