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
import signal
import subprocess
from typing import Any

from merceka_core._env import scrubbed_env

__all__ = [
  "claude_command",
  "claude_env",
  "claude_stream_text_delta",
  "codex_env",
  "codex_exec_command",
  "is_claude_result_event",
  "run_cli",
  "scrubbed_env",
  "signal_process_group",
]

# Read-only tools that locked-down Claude sessions confine to the working directories.
_FENCED_READ_TOOLS = frozenset({"Read", "Grep", "Glob"})


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
  """Build a `claude -p` command. The prompt is passed on stdin by the caller.

  Without ``accept_edits`` (plain text calls and read-only agents), the session is
  locked down:

  - ``allowed_tools`` is the complete tool set (``--tools``). An empty list means no
    tools at all.
  - ``--permission-mode dontAsk`` denies anything that would prompt, instead of
    inheriting the user's ``defaultMode``.
  - Read, Grep and Glob are not pre-approved. Under ``dontAsk`` they run inside the
    working directory and the ``--add-dir`` directories, and are denied everywhere
    else, so a read-only session cannot open ``~/.ssh`` or a ``.env``. Other listed
    tools are pre-approved as given.
  - Only user-level settings load, so project hooks, env blocks and permission rules
    from the working directory don't apply.
  - No MCP servers connect.

  ``--allowedTools`` on its own only pre-approves tools; it never removed the others.

  With ``accept_edits`` (write agents), ``allowed_tools`` is pre-approved under
  ``acceptEdits`` and the rest of the session keeps Claude Code's defaults.
  """
  cmd = [binary, "-p", "--model", model]
  if stream:
    cmd.extend([
      "--output-format", "stream-json",
      "--verbose",
      "--include-partial-messages",
    ])
  if accept_edits:
    cmd.extend(["--permission-mode", "acceptEdits"])
    pre_approved = list(allowed_tools)
  else:
    cmd.extend([
      "--permission-mode", "dontAsk",
      "--setting-sources", "user",
      "--strict-mcp-config",
      "--disallowedTools", "mcp__*",
      # --tools takes bare names; a scoped rule such as Bash(git log *) keeps its
      # scope in --allowedTools below.
      "--tools", ",".join(dict.fromkeys(tool.split("(", 1)[0] for tool in allowed_tools)),
    ])
    pre_approved = [tool for tool in allowed_tools if tool not in _FENCED_READ_TOOLS]
  if system_prompt:
    # --append-system-prompt (not --system-prompt): appends to Claude Code's
    # default prompt instead of replacing it. Full-replace strips the dynamic
    # working-directory/env context, so the model stops knowing its cwd already
    # IS the book dir — it reads the sibling books/ tree and asks "which book?",
    # never reads a file, emits no citation markers. All callers (book-chat,
    # content-chat, enrichment WRITE) supply add_dirs and want that cwd context.
    cmd.extend(["--append-system-prompt", system_prompt])
  for d in add_dirs:
    cmd.extend(["--add-dir", str(d)])
  if pre_approved:
    cmd.extend(["--allowedTools", ",".join(pre_approved)])
  return cmd


def claude_env() -> dict[str, str]:
  """Environment for Claude CLI runs: credentials withheld, and the API key
  blanked so the CLI uses subscription auth instead of accidental API billing.
  ``CLAUDE_CODE_OAUTH_TOKEN`` is the subscription credential, so it is kept."""
  return scrubbed_env(keep=("CLAUDE_CODE_OAUTH_TOKEN",), ANTHROPIC_API_KEY="")


def codex_env() -> dict[str, str]:
  """Environment for Codex CLI runs: credentials withheld, so the CLI uses its
  ChatGPT login (``~/.codex/auth.json``) instead of an inherited API key."""
  return scrubbed_env()


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


# How long a timed-out CLI gets to exit after SIGTERM before its group is killed.
_TIMEOUT_GRACE_S = 5.0


def run_cli(
  cmd: list[str],
  *,
  input: str | None = None,
  timeout: float | None = None,
  env: dict[str, str] | None = None,
  cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
  """``subprocess.run(cmd, capture_output=True, text=True, ...)`` for provider CLIs.

  The CLI leads a new process group, and a timeout stops that whole group.
  ``subprocess.run`` kills only the direct child: codex and pi are node wrappers
  around the real worker, and grok runs tools in subprocesses, so those kept
  running after the caller gave up. Raises ``subprocess.TimeoutExpired`` as
  ``subprocess.run`` does.
  """
  with subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE if input is not None else None,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env=env,
    cwd=cwd,
    start_new_session=True,
  ) as process:
    try:
      stdout, stderr = process.communicate(input, timeout=timeout)
    except subprocess.TimeoutExpired:
      signal_process_group(process, signal.SIGTERM)
      try:
        process.wait(timeout=_TIMEOUT_GRACE_S)
      except subprocess.TimeoutExpired:
        pass
      signal_process_group(process, signal.SIGKILL)
      process.wait()
      raise
    except BaseException:
      signal_process_group(process, signal.SIGKILL)
      process.wait()
      raise
  return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)


def signal_process_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
  """Send ``sig`` to the process group ``process`` leads (see ``start_new_session``)."""
  try:
    os.killpg(process.pid, sig)
  except ProcessLookupError:
    pass  # The group is gone.
  except PermissionError:
    # macOS refuses to signal a group whose only member is an unreaped zombie.
    if process.poll() is None:
      process.send_signal(sig)
