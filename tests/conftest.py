"""Suite-wide isolation.

Runs before any merceka_core import:

- ``.env`` loading is disabled and provider keys are blanked, so a missing mock
  fails instead of making a paid call with a developer's real keys.
- Every test gets its own cost ledger, so the suite never writes rows into
  ``~/.merceka/costs.jsonl``. Each full run used to append 15 to 17.
- Non-integration tests cannot open network connections.
- Non-integration tests cannot launch the real claude/codex/pi/grok/gemini
  CLIs: fakes that record the call come first on PATH, and a test that reaches
  one fails. Patching the wrong seam (``subprocess.run`` after the code moved
  to ``Popen``) once started real, logged-in CLI sessions.
"""

import os
import socket

import pytest

_PROVIDER_CLIS = ("claude", "codex", "pi", "grok", "gemini")

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
for _name in (
  "OPENROUTER_API_KEY",
  "OPENAI_API_KEY",
  "GOOGLE_API_KEY",
  "GEMINI_API_KEY",
  "FAL_KEY",
  "ANTHROPIC_API_KEY",
  "XAI_API_KEY",
):
  os.environ[_name] = ""


@pytest.fixture(autouse=True)
def _isolated_cost_ledger(tmp_path, monkeypatch):
  monkeypatch.setenv("MERCEKA_COST_LEDGER", str(tmp_path / "costs.jsonl"))


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
  if request.node.get_closest_marker("integration"):
    return

  def refuse(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6):
      raise RuntimeError(f"unit tests must not open network connections (tried {address!r})")
    return original(sock, address)

  original = socket.socket.connect
  monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture(scope="session")
def _cli_tripwire(tmp_path_factory):
  """Fake provider CLIs that append their argv to a log and exit 97."""
  bin_dir = tmp_path_factory.mktemp("cli-tripwire")
  log = bin_dir / "invocations.log"
  for name in _PROVIDER_CLIS:
    script = bin_dir / name
    script.write_text(f'#!/bin/sh\necho "{name} $*" >> "{log}"\nexit 97\n')
    script.chmod(0o755)
  return bin_dir, log


@pytest.fixture(autouse=True)
def _no_real_provider_cli(request, monkeypatch, _cli_tripwire):
  if request.node.get_closest_marker("integration"):
    yield
    return
  bin_dir, log = _cli_tripwire
  monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
  before = log.stat().st_size if log.exists() else 0
  yield
  if log.exists() and log.stat().st_size > before:
    with log.open() as fh:
      fh.seek(before)
      calls = fh.read().strip()
    pytest.fail(f"test launched a provider CLI instead of a fake: {calls}")
