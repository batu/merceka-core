"""Suite-wide isolation.

Runs before any merceka_core import:

- ``.env`` loading is disabled and provider keys are blanked, so a missing mock
  fails instead of making a paid call with a developer's real keys.
- Every test gets its own cost ledger, so the suite never writes rows into
  ``~/.merceka/costs.jsonl``. Each full run used to append 15 to 17.
- Non-integration tests cannot open network connections.
"""

import os
import socket

import pytest

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
