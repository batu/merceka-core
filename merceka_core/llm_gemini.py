"""Gemini video understanding and search-grounded generation.

Module-level functions consumed by LLM.generate_with_video /
agenerate_with_video and by downstream callers (slab) via
``generate_with_search_grounding``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from merceka_core.errors import (
  VideoBackendError,
  VideoNotFoundError,
  VideoUploadError,
)
from merceka_core.messages import OutputSchema
from merceka_core.retry import (
  _RETRY_MAX_ATTEMPTS,
  _RETRY_STATUS_CODES,
  _retry_delay,
)

_logger = logging.getLogger(__name__)


def _gemini_client():
  """Construct a google-genai Client lazily (SDK import is heavy)."""
  from google import genai
  from google.genai import types

  # The SDK picks up GOOGLE_API_KEY or GEMINI_API_KEY automatically.
  # http_options.timeout is in milliseconds.
  return genai.Client(http_options=types.HttpOptions(timeout=600_000))


def _gemini_poll_until_active(client, file_obj, timeout_s: float, poll_interval_s: float):
  """Block until ``file_obj.state.name == 'ACTIVE'`` or raise.

  Raises:
    VideoUploadError: On ``FAILED`` or timeout.
  """
  deadline = time.monotonic() + timeout_s
  current = file_obj
  while True:
    state = getattr(current, "state", None)
    state_name = getattr(state, "name", None) or str(state)
    if state_name == "ACTIVE":
      return current
    if state_name == "FAILED":
      raise VideoUploadError(f"Gemini file upload FAILED: {current.name}")
    if time.monotonic() >= deadline:
      raise VideoUploadError(
        f"Gemini file upload did not reach ACTIVE within {timeout_s}s "
        f"(current state={state_name}, name={current.name})"
      )
    time.sleep(poll_interval_s)
    current = client.files.get(name=current.name)


def _build_video_config(max_tokens=None, system_prompt: str = "", **extra):
  """Translate common slab kwargs into a google-genai GenerateContentConfig.

  google-genai rejects unknown top-level kwargs on ``generate_content``
  (e.g. ``max_tokens``) — they have to ride on ``config``. This helper
  keeps callers in merceka-land portable to both the litellm-style
  argument names they already use and the SDK-native shape.
  """
  from google.genai import types

  cfg: dict = {}
  if max_tokens:
    cfg["max_output_tokens"] = int(max_tokens)
  if system_prompt:
    cfg["system_instruction"] = system_prompt
  # Passthrough known config fields the caller may want to set directly.
  for key in (
    "temperature", "top_p", "top_k",
    "stop_sequences", "response_mime_type", "response_schema",
    "safety_settings",
  ):
    if key in extra:
      cfg[key] = extra.pop(key)
  if not cfg:
    return None, extra
  return types.GenerateContentConfig(**cfg), extra


def _gemini_video_call(
  llm,  # LLM instance (forward-decl to avoid circular self-ref in helper)
  message: str,
  video_paths,
  *,
  timeout_s: float,
  poll_interval_s: float,
  **kwargs,
) -> str | OutputSchema:
  """Upload, poll, generate, delete. Blocking."""
  # Normalize to list of Path.
  if isinstance(video_paths, (str, Path)):
    paths = [Path(video_paths)]
  else:
    paths = [Path(p) for p in video_paths]

  for p in paths:
    if not p.exists():
      raise VideoNotFoundError(f"Video not found: {p}")

  client = _gemini_client()
  model_alias = llm.model_name.removeprefix("gemini/")

  # Extract caller kwargs that google-genai doesn't accept as top-level.
  config, remaining_kwargs = _build_video_config(
    max_tokens=kwargs.pop("max_tokens", None),
    system_prompt=llm.system_prompt,
    **kwargs,
  )

  uploaded = []
  try:
    for p in paths:
      try:
        file_obj = client.files.upload(file=str(p))
      except Exception as exc:  # pragma: no cover — SDK-specific errors.
        raise VideoUploadError(f"upload failed for {p}: {exc}") from exc
      active = _gemini_poll_until_active(client, file_obj, timeout_s, poll_interval_s)
      uploaded.append(active)

    contents = [*uploaded, message]

    # Apply retry around generate_content for transient 5xx/429.
    for attempt in range(_RETRY_MAX_ATTEMPTS):
      try:
        gc_kwargs: dict = {"model": model_alias, "contents": contents, **remaining_kwargs}
        if config is not None:
          gc_kwargs["config"] = config
        response = client.models.generate_content(**gc_kwargs)
        break
      except Exception as exc:  # noqa: BLE001 — bridge to our taxonomy.
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        is_retryable = status in _RETRY_STATUS_CODES or isinstance(
          exc, (ConnectionResetError, ConnectionRefusedError)
        )
        if not is_retryable or attempt == _RETRY_MAX_ATTEMPTS - 1:
          raise VideoBackendError(f"Gemini generate_content failed: {exc}") from exc
        delay = _retry_delay(attempt)
        _logger.warning("Gemini %s, retrying in %.2fs", type(exc).__name__, delay)
        time.sleep(delay)

    text = getattr(response, "text", None) or ""
    return llm._parse_response(text)
  finally:
    for f in uploaded:
      try:
        client.files.delete(name=f.name)
      except Exception:  # pragma: no cover — hygiene, not critical.
        _logger.debug("Gemini file delete failed for %s", getattr(f, "name", "?"))

def _extract_grounding(response) -> dict:
  """Pull grounding metadata from a google-genai response into a plain dict.

  Schema:
    {"queries": list[str],
     "citations": list[{"uri": str, "title": str}],
     "search_entry_point_html": str | None}

  Handles python-genai #802: ``grounding_metadata`` may be absent on the
  first candidate even when searches were performed. Returns empty
  queries/citations so the caller can decide to degrade.
  """
  out: dict = {"queries": [], "citations": [], "search_entry_point_html": None}
  candidates = getattr(response, "candidates", None) or []
  if not candidates:
    return out
  gm = getattr(candidates[0], "grounding_metadata", None)
  if gm is None:
    return out
  queries = getattr(gm, "web_search_queries", None) or []
  out["queries"] = [str(q) for q in queries]
  chunks = getattr(gm, "grounding_chunks", None) or []
  citations = []
  for chunk in chunks:
    web = getattr(chunk, "web", None)
    if web is not None:
      citations.append({
        "uri": str(getattr(web, "uri", "") or ""),
        "title": str(getattr(web, "title", "") or ""),
      })
  out["citations"] = citations
  sep = getattr(gm, "search_entry_point", None)
  if sep is not None:
    out["search_entry_point_html"] = getattr(sep, "rendered_content", None)
  return out


def _generate_with_search_grounding_sync(
  *,
  prompt: str,
  system_prompt: str,
  model: str,
  max_tokens: int,
  timeout_s: float,
) -> tuple[str, dict]:
  """Blocking impl; ``generate_with_search_grounding`` wraps this in a thread."""
  from google.genai import types

  client = _gemini_client()
  tools = [types.Tool(google_search=types.GoogleSearch())]
  config_kwargs: dict = {"tools": tools}
  if max_tokens:
    config_kwargs["max_output_tokens"] = max_tokens
  if system_prompt:
    config_kwargs["system_instruction"] = system_prompt
  config = types.GenerateContentConfig(**config_kwargs)

  response = None
  for attempt in range(_RETRY_MAX_ATTEMPTS):
    try:
      response = client.models.generate_content(
        model=model, contents=prompt, config=config,
      )
      break
    except Exception as exc:  # noqa: BLE001 — bridge to our taxonomy.
      status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
      is_retryable = status in _RETRY_STATUS_CODES or isinstance(
        exc, (ConnectionResetError, ConnectionRefusedError)
      )
      if not is_retryable or attempt == _RETRY_MAX_ATTEMPTS - 1:
        raise VideoBackendError(
          f"Gemini search-grounded generate_content failed: {exc}"
        ) from exc
      delay = _retry_delay(attempt)
      _logger.warning(
        "Gemini search-grounded %s, retrying in %.2fs", type(exc).__name__, delay,
      )
      time.sleep(delay)

  text = getattr(response, "text", None) or ""
  try:
    grounding = _extract_grounding(response)
  except Exception as exc:  # noqa: BLE001 — never fail the call on metadata parse.
    _logger.warning("Failed to extract grounding metadata: %s", exc)
    grounding = {"queries": [], "citations": [], "search_entry_point_html": None}
  return text, grounding


async def generate_with_search_grounding(
  *,
  prompt: str,
  system_prompt: str = "",
  model: str = "gemini-2.5-pro",
  max_tokens: int = 6000,
  timeout_s: float = 120.0,
) -> tuple[str, dict]:
  """Gemini generate_content with Google-Search grounding.

  Returns ``(raw_text, grounding_dict)`` where ``grounding_dict`` has
  keys ``queries``, ``citations``, ``search_entry_point_html``.
  When the python-genai SDK omits ``grounding_metadata`` (issue #802),
  the returned lists are empty — the caller is expected to degrade.

  Args:
    prompt: User prompt.
    system_prompt: Optional system instruction.
    model: Gemini model ID (without ``gemini/`` prefix).
    max_tokens: Output cap. ``0`` disables the cap.
    timeout_s: Reserved; the underlying client uses its own timeout.

  Raises:
    VideoBackendError: On non-retryable 5xx / persistent transport errors.
  """
  import asyncio
  return await asyncio.to_thread(
    _generate_with_search_grounding_sync,
    prompt=prompt,
    system_prompt=system_prompt,
    model=model,
    max_tokens=max_tokens,
    timeout_s=timeout_s,
  )
