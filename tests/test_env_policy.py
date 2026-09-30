"""Environment policy: the library never re-injects keys a caller blanked, loads only
its own provider keys from .env, and CLI children never inherit credentials."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from merceka_core import _env


@pytest.fixture
def fake_dotenv(tmp_path, monkeypatch):
  """Point the package .env lookup at a temp file and clear the lookup cache."""
  path = tmp_path / ".env"
  path.write_text(
    "OPENROUTER_API_KEY=sk-or-from-file\n"
    "GOOGLE_API_KEY=g-from-file\n"
    "UNRELATED_SERVICE_PASSWORD=hunter2\n"
    "TELEGRAM_BOT_TOKEN=bot-from-file\n"
    "PIN=1234\n"
  )
  monkeypatch.setattr(_env, "_find_dotenv", lambda start: path)
  _env._dotenv_values.cache_clear()
  for name in ("OPENROUTER_API_KEY", "GOOGLE_API_KEY", "UNRELATED_SERVICE_PASSWORD",
               "TELEGRAM_BOT_TOKEN", "PIN", "PYTHON_DOTENV_DISABLED"):
    monkeypatch.delenv(name, raising=False)
  yield path
  _env._dotenv_values.cache_clear()


@pytest.mark.usefixtures("fake_dotenv")
def test_loads_only_provider_keys():
  _env.load_provider_keys()
  assert os.environ["OPENROUTER_API_KEY"] == "sk-or-from-file"
  assert os.environ["GOOGLE_API_KEY"] == "g-from-file"
  # Non-provider secrets in the same file stay out of the process environment.
  assert "UNRELATED_SERVICE_PASSWORD" not in os.environ
  assert "TELEGRAM_BOT_TOKEN" not in os.environ
  assert "PIN" not in os.environ


@pytest.mark.usefixtures("fake_dotenv")
def test_blank_value_disables_provider(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "")
  _env.load_provider_keys()
  assert os.environ["OPENROUTER_API_KEY"] == ""


@pytest.mark.usefixtures("fake_dotenv")
def test_existing_value_wins(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-from-env")
  _env.load_provider_keys()
  assert os.environ["OPENROUTER_API_KEY"] == "sk-or-from-env"


@pytest.mark.usefixtures("fake_dotenv")
def test_python_dotenv_disabled_skips_file(monkeypatch):
  monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
  _env.load_provider_keys()
  assert "OPENROUTER_API_KEY" not in os.environ


@pytest.mark.usefixtures("fake_dotenv")
def test_scrubbed_env_drops_credentials_and_dotenv_names(monkeypatch):
  monkeypatch.setenv("PIN", "1234")  # a .env name without a credential-looking segment
  monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws")
  monkeypatch.setenv("DB_PASSWORD", "pw")
  monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")
  monkeypatch.setenv("MONKEY_BUSINESS", "not-a-secret")
  env = _env.scrubbed_env()
  assert "PIN" not in env
  assert "AWS_SECRET_ACCESS_KEY" not in env
  assert "DB_PASSWORD" not in env
  assert env["PYTHON_KEYRING_BACKEND"] == "keyring.backends.null.Keyring"
  assert env["MONKEY_BUSINESS"] == "not-a-secret"
  assert "PATH" in env


@pytest.mark.usefixtures("fake_dotenv")
def test_scrubbed_env_keep_and_overrides(monkeypatch):
  monkeypatch.setenv("XAI_API_KEY", "xai")
  env = _env.scrubbed_env(keep=("XAI_API_KEY",), ANTHROPIC_API_KEY="")
  assert env["XAI_API_KEY"] == "xai"
  assert env["ANTHROPIC_API_KEY"] == ""


def test_import_does_not_reinject_blanked_keys(tmp_path):
  """Regression for the 2026-09-30 review: a blanked key must survive import."""
  script = tmp_path / "probe.py"
  script.write_text(
    "import os\n"
    "import merceka_core.llm, merceka_core.llm_gemini\n"
    "print(repr(os.environ.get('OPENROUTER_API_KEY')))\n"
  )
  env = {**os.environ, "OPENROUTER_API_KEY": ""}
  env.pop("PYTHON_DOTENV_DISABLED", None)
  result = subprocess.run(
    [sys.executable, str(script)], capture_output=True, text=True, env=env,
    cwd=Path(__file__).resolve().parents[1],
  )
  assert result.returncode == 0, result.stderr
  assert result.stdout.strip() == "''"
