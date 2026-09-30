"""Gemini video understanding and search-grounded generation.

Module-level functions consumed by LLM.generate_with_video /
agenerate_with_video and by downstream callers (slab) via
``generate_with_search_grounding``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from merceka_core import _env
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

_env.load_provider_keys()  # _gemini_client reads GOOGLE_API_KEY/GEMINI_API_KEY from env


def _gemini_client():
  """Construct a google-genai Client lazily (SDK import is heavy)."""
  from google import genai
  from google.genai import types

  # The SDK picks up GOOGLE_API_KEY or GEMINI_API_KEY automatically.
  # http_options.timeout is in milliseconds.
  return genai.Client(http_options=types.HttpOptions(timeout=600_000))


def _usage_dict(usage_metadata) -> dict:
  """``usage_metadata`` as the REST ``usageMetadata`` dict costs.py prices.

  Validating through the SDK's own type converts the SDK object (or any object
  or dict with its fields) to the camelCase REST names (``promptTokenCount``,
  ``candidatesTokensDetails[].tokenCount``, ...); JSON mode turns the modality
  enums into their ``"IMAGE"``/``"TEXT"`` strings. Anything unreadable yields an
  empty dict, so the call is still counted.
  """
  if usage_metadata is None:
    return {}
  try:
    from google.genai import types

    usage = types.GenerateContentResponseUsageMetadata.model_validate(usage_metadata)
    return usage.model_dump(mode="json", by_alias=True, exclude_none=True)
  except Exception:  # noqa: BLE001 — metering must never fail the metered call.
    return {}


def _record_usage(model: str, response) -> None:
  """Meter one ``generate_content`` response in the cost ledger.

  Called before the response is parsed, so a charged call whose output cannot
  be parsed is still recorded. Gemini states no cost, so ``usd`` is left to
  the rate table.
  """
  from merceka_core import costs as _costs

  _costs.record(
    source="google-direct",
    model=f"google/{model}",
    usage=_usage_dict(getattr(response, "usage_metadata", None)),
    request_id=getattr(response, "response_id", None),
  )


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


# Client errors: the request itself is wrong (bad argument, bad or unauthorised
# key, unknown model), so every retry fails the same way.
_CLIENT_ERROR_STATUS_CODES = frozenset({400, 401, 403, 404})


def _generate_content(client, label: str, **gc_kwargs):
  """``client.models.generate_content`` with the shared retry policy.

  Retries 429/5xx and connection resets. Client errors (400/401/403/404) and a
  TypeError (arguments ``generate_content`` does not accept) raise the terminal
  :class:`VideoUploadError`, because retrying cannot fix them. Everything else,
  including exhausted retries, raises the transient :class:`VideoBackendError`.
  """
  for attempt in range(_RETRY_MAX_ATTEMPTS):
    try:
      return client.models.generate_content(**gc_kwargs)
    except Exception as exc:  # noqa: BLE001 — bridge SDK errors to our taxonomy.
      if isinstance(exc, TypeError):
        raise VideoUploadError(
          f"{label} generate_content rejected its arguments: {exc}"
        ) from exc
      status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
      if status in _CLIENT_ERROR_STATUS_CODES:
        raise VideoUploadError(
          f"{label} generate_content rejected the request ({status}): {exc}"
        ) from exc
      is_retryable = status in _RETRY_STATUS_CODES or isinstance(
        exc, (ConnectionResetError, ConnectionRefusedError)
      )
      if not is_retryable or attempt == _RETRY_MAX_ATTEMPTS - 1:
        raise VideoBackendError(f"{label} generate_content failed: {exc}") from exc
      delay = _retry_delay(attempt)
      _logger.warning("%s %s, retrying in %.2fs", label, type(exc).__name__, delay)
      time.sleep(delay)
  raise RuntimeError("retry loop exhausted without return")  # unreachable


def _delete_uploads(client, names: list[str]) -> None:
  """Best-effort delete of uploaded files. Never raises: cleanup must not mask
  the error that ended the call."""
  for name in names:
    try:
      client.files.delete(name=name)
    except Exception:  # noqa: BLE001 — hygiene, not critical.
      _logger.warning("Gemini file delete failed for %s", name)


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

  # Enforce structured output at the API level, as the image path does.
  output_schema = getattr(llm, "output_schema", None)
  if output_schema is not None:
    kwargs.setdefault("response_mime_type", "application/json")
    kwargs.setdefault("response_schema", output_schema)
  # Extract caller kwargs that google-genai doesn't accept as top-level.
  config, remaining_kwargs = _build_video_config(
    max_tokens=kwargs.pop("max_tokens", None),
    system_prompt=llm.system_prompt,
    **kwargs,
  )

  uploaded = []  # ACTIVE file objects, in upload order: the generate_content contents
  to_delete: list[str] = []  # every file the Files API accepted, ACTIVE or not
  try:
    for p in paths:
      try:
        file_obj = client.files.upload(file=str(p))
      except Exception as exc:  # pragma: no cover — SDK-specific errors.
        raise VideoUploadError(f"upload failed for {p}: {exc}") from exc
      # Queue the delete before polling, so an upload that never becomes
      # ACTIVE (FAILED or poll timeout) is still removed.
      if file_obj.name:
        to_delete.append(file_obj.name)
      active = _gemini_poll_until_active(client, file_obj, timeout_s, poll_interval_s)
      uploaded.append(active)

    contents = [*uploaded, message]
    gc_kwargs: dict = {"model": model_alias, "contents": contents, **remaining_kwargs}
    if config is not None:
      gc_kwargs["config"] = config
    response = _generate_content(client, "Gemini", **gc_kwargs)

    _record_usage(model_alias, response)
    text = getattr(response, "text", None) or ""
    return llm._parse_response(text)
  finally:
    _delete_uploads(client, to_delete)

_IMAGE_MIME_FALLBACK = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".bmp": "image/bmp",
  ".tiff": "image/tiff",
}


def _gemini_image_call(llm, message: str, resource_path, **kwargs):
  """Image understanding via inline bytes (no Files API ceremony).

  Called by ``LLM.generate_with_resource`` for ``gemini/`` models. Unlike
  the video path, images ride inline on the request, so there is no
  upload/poll/delete lifecycle. Retries transient failures via the shared
  retry policy; raises :class:`VideoUploadError` when the request is rejected
  (400/401/403/404) and :class:`VideoBackendError` (the shared transient Gemini
  error) when retries are exhausted or the failure is unclassified.
  """
  import mimetypes

  from google.genai import types

  path = Path(resource_path)
  if not path.exists():
    raise FileNotFoundError(f"Resource not found: {path}")

  mime_type, _ = mimetypes.guess_type(str(path))
  if mime_type is None:
    mime_type = _IMAGE_MIME_FALLBACK.get(path.suffix.lower(), "application/octet-stream")

  model = llm.model_name.removeprefix("gemini/")
  # Enforce structured output at the API level (parity with the OpenRouter
  # and Ollama branches of generate_with_resource, which send a JSON schema).
  if llm.output_schema is not None:
    kwargs.setdefault("response_mime_type", "application/json")
    kwargs.setdefault("response_schema", llm.output_schema)
  config, remaining_kwargs = _build_video_config(system_prompt=llm.system_prompt, **kwargs)
  part = types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)

  client = _gemini_client()
  # Unknown kwargs reach generate_content and fail loudly (video-path parity).
  gc_kwargs: dict = {"model": model, "contents": [part, message], **remaining_kwargs}
  if config is not None:
    gc_kwargs["config"] = config
  response = _generate_content(client, "Gemini image", **gc_kwargs)

  _record_usage(model, response)
  text = getattr(response, "text", None) or ""
  if not text and llm.output_schema is not None:
    # Blocked/empty response would surface as a confusing ValidationError.
    raise VideoBackendError(
      "Gemini image call returned an empty response (safety-filtered or "
      "truncated) — cannot parse into the requested output_schema.")
  return llm._parse_response(text)


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

  response = _generate_content(
    client, "Gemini search-grounded", model=model, contents=prompt, config=config,
  )

  _record_usage(model, response)
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
    VideoUploadError: The request was rejected (400/401/403/404).
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
