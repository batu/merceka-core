"""Process plumbing shared by the CLI agent providers (claude, codex, pi).

Each provider keeps its own command line, prompt and event parsing. The
lifecycle code that is identical across them lives here.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from merceka_core.agent import RawProviderEvent


def raw_event_from_line(line: str, provider: str) -> RawProviderEvent:
  """One JSON line of CLI output as a raw event; unparseable lines are kept as such."""
  try:
    payload: Any = json.loads(line)
  except json.JSONDecodeError as exc:
    return RawProviderEvent(
      provider=provider,
      event_type="malformed_json",
      payload={"line": line, "error": str(exc)},
    )
  event_type = str(payload.get("type", "raw")) if isinstance(payload, dict) else "raw"
  return RawProviderEvent(provider=provider, event_type=event_type, payload=payload)


def raw_events_from_stdout(stdout: str, provider: str) -> list[RawProviderEvent]:
  return [raw_event_from_line(line, provider) for line in stdout.splitlines() if line.strip()]


def terminate_process(process: subprocess.Popen[str]) -> None:
  process.terminate()
  process.wait()


def close_pipe(pipe: Any) -> None:
  if pipe is not None:
    pipe.close()
