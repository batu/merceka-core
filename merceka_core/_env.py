"""Environment policy: which ``.env`` entries the library loads, and what CLI children see.

Importing ``merceka_core.llm`` used to call ``dotenv.load_dotenv()``, which copied
*every* entry of the nearest ``.env`` into ``os.environ``. That included unrelated
credentials, not only provider keys. Every subprocess the consumer
started inherited them, including CLI agents that can run shell commands.

Now:

- :func:`load_provider_keys` loads only the credentials this library reads (see
  ``PROVIDER_KEYS``), and only when they are absent from the environment. Consumers
  still see these keys in ``os.environ`` after importing the LLM modules; slab,
  videototext and the fabrika level editor check them there.
- To turn a provider off, set its variable to an empty string, or set
  ``PYTHON_DOTENV_DISABLED=1`` to skip ``.env`` loading entirely. Unsetting a
  variable (``env -u``) does not work: an absent key is filled from ``.env``.
- :func:`scrubbed_env` builds the environment for CLI subprocesses. It drops
  credential-looking variables, and every name defined in that ``.env``.
"""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Iterable
from pathlib import Path

__all__ = ["PROVIDER_KEYS", "load_provider_keys", "scrubbed_env"]

# Credentials and settings the library itself reads from the environment.
# ANTHROPIC_API_KEY is deliberately absent: exporting it would switch Claude CLI
# children from subscription auth to API billing.
PROVIDER_KEYS = (
  "OPENROUTER_API_KEY",
  "OPENROUTER_HTTP_REFERER",
  "OPENROUTER_X_TITLE",
  "OPENAI_API_KEY",
  "GOOGLE_API_KEY",
  "GEMINI_API_KEY",
  "FAL_KEY",
  "MERCEKA_FORCE_OPENROUTER",
)

_TRUTHY = {"1", "true", "t", "yes", "y"}
# Underscore-delimited name segments that mark a credential. Segment matching
# keeps GNOME_KEYRING_CONTROL or PYTHON_KEYRING_BACKEND, which CLI tools need.
_SECRET_SEGMENT = re.compile(
  r"(?:^|_)(?:KEY|KEYS|TOKEN|TOKENS|SECRET|SECRETS|PASSWORD|PASSWD|PASS|CREDENTIAL|CREDENTIALS)(?:_|$)"
)


def load_provider_keys() -> None:
  """Copy ``PROVIDER_KEYS`` from the package ``.env`` into ``os.environ`` when absent."""
  if os.environ.get("PYTHON_DOTENV_DISABLED", "").casefold() in _TRUTHY:
    return
  values = _dotenv_values()
  for name in PROVIDER_KEYS:
    value = values.get(name)
    if name not in os.environ and value is not None:
      os.environ[name] = value


def scrubbed_env(keep: Iterable[str] = (), **overrides: str) -> dict[str, str]:
  """Copy of ``os.environ`` for a CLI subprocess, with credentials withheld.

  Drops variables whose names contain a credential segment (``*_API_KEY``,
  ``*_TOKEN``, ``*_PASSWORD``, ...) and every name defined in the package ``.env``,
  unless the name is listed in ``keep``. ``overrides`` are applied last.
  """
  kept = set(keep)
  withheld = set(_dotenv_values())
  env = {
    name: value
    for name, value in os.environ.items()
    if name in kept or (name not in withheld and not _SECRET_SEGMENT.search(name.upper()))
  }
  env.update(overrides)
  return env


@functools.cache
def _dotenv_values() -> dict[str, str]:
  path = _find_dotenv(Path(__file__).resolve().parent)
  if path is None:
    return {}
  from dotenv import dotenv_values

  return {name: value for name, value in dotenv_values(path).items() if value is not None}


def _find_dotenv(start: Path) -> Path | None:
  """Nearest ``.env`` in ``start`` or a parent: the file ``load_dotenv()`` used to load."""
  for directory in (start, *start.parents):
    candidate = directory / ".env"
    if candidate.is_file():
      return candidate
  return None
