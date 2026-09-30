"""Load and run ollama models"""

__all__ = [
  "Tool",
  "list_local_models",
  "create_message",
  "create_message_with_resource",
  "create_ollama_vision_message",
  "tool_from_callable",
  "OutputSchema",
  "LLM",
  "generate_with_search_grounding",
]

import contextlib
import json
import logging
import os
import subprocess
import tempfile
import httpx
import time
import urllib.error

from merceka_core import _env

_env.load_provider_keys()

_logger = logging.getLogger(__name__)

CLAUDE_CLI_TIMEOUT = 120  # seconds
OPENROUTER_HTTP_TIMEOUT = 120  # seconds, when neither LLM(timeout=) nor timeout= is given
# Streaming Claude CLI teardown. A child whose output ended gets the grace period
# to exit on its own; SIGTERM and then SIGKILL follow, each wait bounded, so an
# abandoned or stuck stream can never block its consumer's teardown.
_STREAM_EXIT_GRACE_S = 5.0
_STREAM_SIGNAL_WAIT_S = 5.0
_STREAM_STDERR_TAIL = 2000  # characters of CLI stderr carried on a stream failure
_STREAM_STDERR_WAIT_S = 1.0


from typing import Optional
from ollama import list as ollama_list


def list_local_models():
  """List local models."""
  return [m.model for m in ollama_list()["models"]]


from tqdm import tqdm
from ollama import pull


def _download_model(model_name: str):
  """Download a model from ollama."""
  current_digest, bars = "", {}
  for progress in pull(model_name, stream=True):
    digest = progress.get("digest", "")
    if digest != current_digest and current_digest in bars:
      bars[current_digest].close()

    if not digest:
      print(progress.get("status"))
      continue

    if digest not in bars and (total := progress.get("total")):
      bars[digest] = tqdm(total=total, desc=f"pulling {digest[7:19]}", unit="B", unit_scale=True)

    if completed := progress.get("completed"):
      bars[digest].update(completed - bars[digest].n)

    current_digest = digest


from ollama import chat as ollama_chat


from pathlib import Path


from typing import Callable, cast


from pydantic import BaseModel


from ollama import ChatResponse, ResponseError
from urllib.request import Request, urlopen


from merceka_core.messages import (  # noqa: E402, F401 — re-exported for back-compat
  OutputSchema,
  Tool,
  _openrouter_response_format,
  _parse_param_docs,
  _python_type_to_json,
  _schema_name,
  create_message,
  create_message_with_resource,
  create_ollama_vision_message,
  tool_from_callable,
)
from merceka_core import _cli
from merceka_core.errors import (  # noqa: F401 — VideoNotFoundError/VideoUploadError re-exported
  LLMResponseError,
  VideoBackendError,
  VideoNotFoundError,
  VideoUploadError,
)

# Retry policy for transient HTTP failures on the cloud path.
# Backend decisions returned by LLM._select_backend().
_BACKEND_CLAUDE = "claude"
_BACKEND_CODEX = "codex"
_BACKEND_TOOLS_FALLBACK = "tools_fallback"  # CLI provider + Python tools + fallback set
_BACKEND_TOOL_LOOP = "tool_loop"
_BACKEND_OPENROUTER = "openrouter"
_BACKEND_LOCAL = "local"

from merceka_core.retry import (  # noqa: F401 — re-exported for back-compat
  _RETRY_BASE_DELAY,
  _RETRY_HTTPX_ERRORS,
  _RETRY_MAX_ATTEMPTS,
  _RETRY_MAX_DELAY,
  _RETRY_STATUS_CODES,
  _retry_delay,
  _retry_after_seconds,
  _urlerror_never_sent,
)

# Primary failures that send generate/agenerate to the fallback: transport,
# CLI and provider-side errors. Ollama reports server-side failures (model
# load, out of memory, unknown model) as ResponseError.
_FALLBACK_ERRORS = (
  subprocess.TimeoutExpired,
  subprocess.CalledProcessError,
  FileNotFoundError,
  ConnectionError,
  OSError,
  httpx.HTTPError,
  urllib.error.URLError,
  VideoBackendError,
  ResponseError,
  LLMResponseError,
)

# Per-call kwargs a fallback on another transport must not receive.
# Read only by the CLI transports (_claude_call/_codex_call).
_CLI_KWARGS = frozenset({"timeout", "images"})
# Accepted by ollama.chat() besides the model/messages/tools/think/format that
# _local_call sets itself.
_OLLAMA_KWARGS = frozenset({"options", "keep_alive", "stream", "logprobs", "top_logprobs"})
# The Ollama kwargs OpenRouter has no counterpart for.
_OLLAMA_ONLY_KWARGS = frozenset({"options", "keep_alive"})
# What a Gemini fallback accepts (llm_gemini._build_video_config), and the part of
# it OpenRouter and Ollama have no counterpart for.
_GEMINI_KWARGS = frozenset({
  "max_tokens", "temperature", "top_p", "top_k",
  "stop_sequences", "response_mime_type", "response_schema", "safety_settings",
})
_GEMINI_ONLY_KWARGS = frozenset({
  "stop_sequences", "response_mime_type", "response_schema", "safety_settings",
})
# OpenRouter sampling kwargs and their names in Ollama's ``options``.
_OLLAMA_OPTIONS = {
  "temperature": "temperature",
  "top_p": "top_p",
  "top_k": "top_k",
  "seed": "seed",
  "max_tokens": "num_predict",
  "frequency_penalty": "frequency_penalty",
  "presence_penalty": "presence_penalty",
  "repetition_penalty": "repeat_penalty",
}


def _stop_stream_process(process: subprocess.Popen, *, finished: bool) -> bool:
  """Reap a streaming CLI child without ever blocking indefinitely.

  A child whose output ended (``finished``) gets a grace period to exit on its
  own; an abandoned or failed stream is terminated at once. SIGTERM is followed
  by SIGKILL if it is ignored. Returns True when the child had to be signalled.
  """
  if finished:
    try:
      process.wait(timeout=_STREAM_EXIT_GRACE_S)
      return False
    except subprocess.TimeoutExpired:
      pass
  if process.poll() is not None:
    return False
  process.terminate()
  try:
    process.wait(timeout=_STREAM_SIGNAL_WAIT_S)
  except subprocess.TimeoutExpired:
    process.kill()
    try:
      process.wait(timeout=_STREAM_SIGNAL_WAIT_S)
    except subprocess.TimeoutExpired:
      _logger.error("Claude CLI pid %s did not exit after SIGKILL", process.pid)
  return True


def _read_pipe_tail(pipe, limit: int) -> str:
  """The last ``limit`` characters left in an exited child's output ``pipe``.

  Bounded by ``_STREAM_STDERR_WAIT_S``: a grandchild that inherited the pipe can
  hold it open after the child exits, and an unbounded read would then block.
  """
  import select

  chunks: list[bytes] = []
  try:
    fd = pipe.fileno()
    deadline = time.monotonic() + _STREAM_STDERR_WAIT_S
    while (remaining := deadline - time.monotonic()) > 0:
      ready, _, _ = select.select([fd], [], [], remaining)
      if not ready:
        break
      data = os.read(fd, 65536)
      if not data:
        break
      chunks.append(data)
  except (OSError, ValueError):
    pass  # the detail is best-effort; the failure is raised regardless
  return b"".join(chunks).decode("utf-8", errors="replace")[-limit:]


def _decode_tool_arguments(msg: dict) -> None:
  """Decode JSON-string tool-call arguments in place, where they are valid JSON.

  Invalid arguments stay a string; the tool loop reports them to the model.
  """
  for tc in msg.get("tool_calls") or []:
    args = tc["function"].get("arguments")
    if isinstance(args, str):
      with contextlib.suppress(json.JSONDecodeError):
        tc["function"]["arguments"] = json.loads(args)


def _tool_arguments(tool_call: dict) -> tuple[dict, str | None]:
  """A tool call's arguments as a dict, or an error message for the model.

  Models sometimes emit arguments that are not a JSON object (truncated,
  single-quoted). The message becomes the tool result, so the model can retry,
  instead of the whole call failing.
  """
  fn = tool_call["function"]
  args = fn.get("arguments")
  if args is None or args == "":
    return {}, None
  if isinstance(args, str):
    try:
      args = json.loads(args)
    except json.JSONDecodeError as exc:
      return {}, f"Error: arguments for {fn['name']} are not valid JSON ({exc}): {args[:200]}"
  if not isinstance(args, dict):
    return {}, f"Error: arguments for {fn['name']} must be a JSON object, got {args!r:.200}"
  return args, None


def _claude_result_error(event: dict) -> str | None:
  """The failure message of a stream-json ``result`` event, or None on success.

  Failed runs set ``is_error`` (usage limit, auth, API errors carry the message in
  ``result``) or use an ``error_*`` subtype (max turns, execution errors, which
  list messages in ``errors``).
  """
  subtype = str(event.get("subtype") or "")
  if not event.get("is_error") and not subtype.startswith("error"):
    return None
  errors = event.get("errors")
  detail = event.get("result") or ("; ".join(map(str, errors)) if errors else "")
  return f"Claude CLI reported an error ({subtype or 'unknown'}): {detail or 'no detail'}"


class LLM:
  """A class for interacting with an LLM."""

  def __init__(
    self,
    model_name: str,  # The name of the model to use
    system_prompt: str = "",  # The system prompt to use.
    think: Optional[bool] = None,  # Whether to enable thinking mode
    output_schema: Optional[type[BaseModel]] = None,  # Schema for structured output
    tools: list[Tool] | None = None,  # Tool functions for agentic calling
    max_tool_rounds: int = 10,  # Max iterations of the tool loop
    fallback: Optional[str] = None,  # Fallback model if primary fails
    add_dirs: list[str] | None = None,  # Directories Claude Code can access (--add-dir)
    allowed_tools: list[str] | None = None,  # Claude Code native tools (--allowedTools)
    timeout: int | None = None,  # Default subprocess timeout (seconds) for CLI providers; per-call timeout= kwarg still wins
  ):
    if tools and output_schema:
      raise ValueError("Cannot use both tools and output_schema at the same time")

    self.model_name = model_name
    self.system_prompt = system_prompt
    self.output_schema = output_schema
    self.messages: list[dict] = [create_message(system_prompt, "system")]
    self.think = think
    self.max_tool_rounds = max_tool_rounds
    self.fallback = fallback
    self.timeout = timeout
    self.use_claude = model_name.startswith("claude/")
    self.use_codex = model_name.startswith("codex/")
    self.use_gemini = model_name.startswith("gemini/")
    self.use_openrouter = (
      not self.use_claude
      and not self.use_codex
      and not self.use_gemini
      and "openrouter" in model_name
    )
    self.add_dirs = add_dirs or []
    self.allowed_tools = allowed_tools or []

    # Process tools into schemas and handlers
    self._original_tools = tools
    self._tool_schemas: list[dict] = []
    self._tool_handlers: dict[str, Callable] = {}
    if tools:
      for tool in tools:
        if isinstance(tool, tuple):
          schema, handler = tool
          self._tool_schemas.append(schema)
          self._tool_handlers[schema["function"]["name"]] = handler
        else:
          schema = tool_from_callable(tool)
          self._tool_schemas.append(schema)
          self._tool_handlers[tool.__name__] = tool

    if (
      not self.use_openrouter and not self.use_claude and not self.use_codex and not self.use_gemini
    ):
      self._verify()

  def _fallback_llm(self, model_name: Optional[str] = None) -> "LLM":
    """Construct a fallback LLM preserving the full configuration of this one."""
    return LLM(
      model_name or self.fallback,
      system_prompt=self.system_prompt,
      think=self.think,
      output_schema=self.output_schema,
      tools=self._original_tools,
      max_tool_rounds=self.max_tool_rounds,
      add_dirs=self.add_dirs,
      allowed_tools=self.allowed_tools,
      timeout=self.timeout,
    )

  def _select_backend(self) -> str:
    """Decide which backend serves plain generate/agenerate for this config.

    Single source of truth for dispatch: both the sync and async ladders map
    the returned constant to a transport call, so they cannot diverge.
    Raises eagerly for configurations that have no working backend.
    """
    if self.use_gemini:
      raise ValueError(
        f"{self.model_name!r} is a Gemini model: plain generate/chat is not supported. "
        "Use generate_with_video/agenerate_with_video or generate_with_search_grounding, "
        "or route text through an openrouter/ model."
      )
    if (self.use_claude or self.use_codex) and self._tool_schemas:
      # CLI providers can't run Python tool callables in-process, but both
      # forward allowed_tools to their native tool systems.
      if self.allowed_tools:
        return _BACKEND_CLAUDE if self.use_claude else _BACKEND_CODEX
      if self.fallback:
        return _BACKEND_TOOLS_FALLBACK
      raise ValueError(
        f"{self.model_name!r} cannot run Python tool callables. Either pass "
        "allowed_tools= (native CLI tools), set fallback= to a "
        "tool-capable model, or drop tools=."
      )
    if self.use_claude:
      return _BACKEND_CLAUDE
    if self.use_codex:
      return _BACKEND_CODEX
    if self._tool_schemas:
      return _BACKEND_TOOL_LOOP
    if self.use_openrouter:
      return _BACKEND_OPENROUTER
    return _BACKEND_LOCAL

  def generate(self, message: str, **kwargs) -> str | OutputSchema:
    """One-shot generation. Does not maintain conversation history."""
    try:
      return self._generate_primary(message, **kwargs)
    except _FALLBACK_ERRORS as e:
      target = self._cascade_target(kwargs)
      if target is None:
        raise
      _logger.warning(
        "Primary LLM failed (%s), falling back to %s", type(e).__name__, self.fallback
      )
      fb, fb_kwargs = target
      return fb.generate(message, **fb_kwargs)

  def _fallback_call(self, kwargs: dict) -> tuple["LLM", dict] | None:
    """The fallback LLM and ``kwargs`` adapted to its transport.

    Per-call kwargs are written for the primary's transport. Ollama rejects
    unknown kwargs with a TypeError and OpenRouter would send them in the request
    body, so the fallback gets only what its transport accepts: CLI models keep
    everything (they read ``timeout``/``images`` and ignore the rest), OpenRouter
    loses the CLI- and Ollama-only kwargs, and Ollama gets its own kwargs with
    sampling parameters moved into ``options``. Returns None when the fallback
    cannot honour the call: ``images`` reach only a codex fallback, and any other
    one would answer without seeing them.
    """
    fb = self._fallback_llm()
    if kwargs.get("images") and not fb.use_codex:
      return None
    if fb.use_claude or fb.use_codex:
      return fb, dict(kwargs)
    if fb.use_gemini:
      return fb, {k: v for k, v in kwargs.items() if k in _GEMINI_KWARGS}
    if fb.use_openrouter:
      dropped = _CLI_KWARGS | _OLLAMA_ONLY_KWARGS | _GEMINI_ONLY_KWARGS
      return fb, {k: v for k, v in kwargs.items() if k not in dropped}
    fb_kwargs = {k: v for k, v in kwargs.items() if k in _OLLAMA_KWARGS}
    options = {_OLLAMA_OPTIONS[k]: v for k, v in kwargs.items() if k in _OLLAMA_OPTIONS}
    if options:
      fb_kwargs["options"] = {**options, **(fb_kwargs.get("options") or {})}
    dropped = sorted(set(kwargs) - set(fb_kwargs) - set(_OLLAMA_OPTIONS))
    if dropped:
      _logger.debug("Not passing %s to Ollama fallback %s", dropped, fb.model_name)
    return fb, fb_kwargs

  def _cascade_target(self, kwargs: dict) -> tuple["LLM", dict] | None:
    """Where generate/agenerate send a call whose primary failed; None re-raises.

    None when no fallback is set, when the fallback already served the call (a
    CLI model with Python tools hands them to the fallback, and a second run
    would repeat its tool calls), or when it cannot honour the call's kwargs.
    """
    if not self.fallback or self._select_backend() == _BACKEND_TOOLS_FALLBACK:
      return None
    return self._fallback_call(kwargs)

  def _tools_fallback_call(self, kwargs: dict) -> tuple["LLM", dict]:
    """The fallback that serves a CLI model's Python tools, with adapted kwargs."""
    _logger.info(
      "%s can't run Python tool callables, using fallback %s", self.model_name, self.fallback
    )
    target = self._fallback_call(kwargs)
    if target is None:
      raise ValueError(
        f"{self.model_name!r} hands its Python tools to fallback {self.fallback!r}, which "
        "cannot receive images=. Use a codex/ fallback, or drop tools= or images=."
      )
    return target

  def _generate_primary(self, message: str, **kwargs) -> str | OutputSchema:
    """Primary generation dispatch."""
    messages = [create_message(self.system_prompt, "system"), create_message(message, "user")]
    response, _ = self._run_backend(self._select_backend(), messages, message, **kwargs)
    return response

  def _run_backend(
    self, backend: str, messages: list[dict], cli_prompt: str, **kwargs,
  ) -> tuple[str | OutputSchema, list[dict] | None]:
    """Map a ``_select_backend()`` decision to its sync transport.

    The one sync decision -> transport map, shared by generate and chat.
    ``messages`` feed the HTTP and Ollama backends; the one-shot CLI backends get
    ``cli_prompt`` instead (their system prompt travels separately). Returns the
    response and, for the Python tool loop, the full message trace (else None).
    """
    if backend == _BACKEND_TOOLS_FALLBACK:
      fb, fb_kwargs = self._tools_fallback_call(kwargs)
      return fb._run_backend(fb._select_backend(), messages, cli_prompt, **fb_kwargs)
    if backend == _BACKEND_CLAUDE:
      return self._claude_call(cli_prompt, **kwargs), None
    if backend == _BACKEND_CODEX:
      return self._codex_call(cli_prompt, **kwargs), None
    if backend == _BACKEND_TOOL_LOOP:
      return self._run_tool_loop(messages, **kwargs)
    if backend == _BACKEND_OPENROUTER:
      return self._cloud_call(messages, **kwargs), None
    return self._local_call(messages, **kwargs), None

  def chat(self, message: str, **kwargs) -> str | OutputSchema:
    """Multi-turn chat. Maintains conversation history."""
    backend = self._select_backend()  # raises before history changes
    self.messages.append(create_message(message, "user"))
    # CLI providers are one-shot: they get the history as text, without the
    # system prompt, which they receive separately.
    history = "\n".join(
      f"{m['role']}: {m['content']}"
      for m in self.messages
      if m.get("content") and m["role"] != "system"
    )
    response, trace = self._run_backend(backend, list(self.messages), history, **kwargs)

    if trace is not None:
      self.messages = trace  # the tool loop's trace ends with the assistant reply
    elif isinstance(response, BaseModel):
      # Schema responses keep .content (or their JSON) in history.
      self.messages.append(create_message(self._response_to_history_content(response), "assistant"))
    else:
      self.messages.append(create_message(response, "assistant"))
    return response

  def generate_with_resource(
    self,
    message: str,
    resource_path: Path | str,
    **kwargs,
  ) -> str | OutputSchema:
    """One-shot generation with an attached file (image/PDF).

    Supports OpenRouter cloud models, Gemini, local Ollama vision models, and
    Codex CLI models for images (``codex exec -i``). Claude CLI is not
    supported (the CLI takes stdin text only). Does not maintain
    conversation history.

    Args:
      message: The text prompt to accompany the resource.
      resource_path: Path to the file (image or PDF).
      **kwargs: Additional args passed to the API.

    Returns:
      Model response as string or OutputSchema.
    """
    try:
      return self._resource_primary(message, resource_path, **kwargs)
    except _FALLBACK_ERRORS as e:
      target = self._resource_fallback(resource_path, kwargs)
      if target is None:
        raise
      _logger.warning(
        "Primary LLM failed (%s), falling back to %s", type(e).__name__, self.fallback
      )
      fb, fb_kwargs = target
      return fb.generate_with_resource(message, resource_path, **fb_kwargs)

  def _resource_fallback(
    self, resource_path: Path | str, kwargs: dict,
  ) -> tuple["LLM", dict] | None:
    """The fallback for a failed resource call, or None to re-raise.

    None without a fallback, when the resource itself is missing (the fallback
    would fail the same way), when the fallback is a Claude CLI model (it takes
    no attachments), or when it cannot honour the call's kwargs.
    """
    if not self.fallback or not Path(resource_path).exists():
      return None
    target = self._fallback_call(kwargs)
    if target is None or target[0].use_claude:
      return None
    return target

  def _resource_primary(
    self, message: str, resource_path: Path | str, **kwargs,
  ) -> str | OutputSchema:
    """generate_with_resource on this model, without the fallback."""
    if self.use_claude:
      raise ValueError(
        "generate_with_resource is not supported for Claude CLI models — "
        "the CLI accepts stdin text only. Use an openrouter, gemini, or ollama model."
      )

    if self.use_codex:
      return self._codex_call(message, images=[self._codex_image(resource_path)], **kwargs)

    if self.use_gemini:
      return _gemini_image_call(self, message, resource_path, **kwargs)

    if self.use_openrouter:
      messages = [
        create_message(self.system_prompt, "system"),
        create_message_with_resource(message, resource_path, "user"),
      ]
      return self._cloud_call(messages, **kwargs)

    # Local Ollama path — use Ollama-native image format.
    messages = [
      create_message(self.system_prompt, "system"),
      create_ollama_vision_message(message, resource_path, "user"),
    ]
    return self._local_call(messages, **kwargs)

  async def agenerate_with_resource(
    self,
    message: str,
    resource_path: Path | str,
    **kwargs,
  ) -> str | OutputSchema:
    """Async one-shot generation with an attached file (image/PDF).

    Mirrors :meth:`generate_with_resource` but runs the blocking calls (Ollama,
    Gemini, Codex CLI) in a worker thread so they don't block the event loop.
    Claude CLI is not supported.
    """
    import asyncio

    try:
      return await self._aresource_primary(message, resource_path, **kwargs)
    except _FALLBACK_ERRORS as e:
      # A local fallback's constructor blocks (Ollama _verify): worker thread.
      target = await asyncio.to_thread(self._resource_fallback, resource_path, kwargs)
      if target is None:
        raise
      _logger.warning(
        "Primary LLM failed (%s), falling back to %s", type(e).__name__, self.fallback
      )
      fb, fb_kwargs = target
      return await fb.agenerate_with_resource(message, resource_path, **fb_kwargs)

  async def _aresource_primary(
    self, message: str, resource_path: Path | str, **kwargs,
  ) -> str | OutputSchema:
    """agenerate_with_resource on this model, without the fallback."""
    import asyncio

    if self.use_claude:
      raise ValueError(
        "agenerate_with_resource is not supported for Claude CLI models — "
        "the CLI accepts stdin text only. Use an openrouter, gemini, or ollama model."
      )

    if self.use_codex:
      image = self._codex_image(resource_path)
      return await asyncio.to_thread(self._codex_call, message, images=[image], **kwargs)

    if self.use_gemini:
      return await asyncio.to_thread(_gemini_image_call, self, message, resource_path, **kwargs)

    if self.use_openrouter:
      messages = [
        create_message(self.system_prompt, "system"),
        create_message_with_resource(message, resource_path, "user"),
      ]
      return await self._acloud_call(messages, **kwargs)

    messages = [
      create_message(self.system_prompt, "system"),
      create_ollama_vision_message(message, resource_path, "user"),
    ]
    return await asyncio.to_thread(self._local_call, messages, **kwargs)

  # --- Raw call methods (return full message dict for tool loop) ---

  def _codex_image(self, resource_path: Path | str) -> str:
    """``resource_path`` as a ``codex exec -i`` image, or ValueError.

    codex attaches images only, so other resources (PDFs) have no route.
    """
    import mimetypes

    mime_type, _ = mimetypes.guess_type(str(resource_path))
    if not (mime_type or "").startswith("image/"):
      raise ValueError(
        f"{self.model_name!r}: codex exec attaches images only (-i), not "
        f"{mime_type or 'unknown type'} files like {Path(resource_path).name!r}. "
        "Use an openrouter or gemini model for other resources."
      )
    return str(resource_path)

  def _local_call_raw(self, messages: list[dict], **kwargs) -> dict:
    """Call local Ollama and return normalized message dict."""
    response: ChatResponse = ollama_chat(
      model=self.model_name,
      think=self.think,
      messages=messages,
      tools=self._tool_schemas or None,
      **kwargs,
    )
    msg = response.message
    # Normalize Ollama ToolCall objects to OpenAI format
    tool_calls = None
    if msg.tool_calls:
      tool_calls = []
      for i, tc in enumerate(msg.tool_calls):
        tool_calls.append(
          {
            "id": f"call_{i}",
            "type": "function",
            "function": {
              "name": tc.function.name,
              "arguments": tc.function.arguments,
            },
          }
        )
    return {
      "role": "assistant",
      "content": msg.content,
      "tool_calls": tool_calls,
    }

  def _cloud_call_raw(self, messages: list[dict], **kwargs) -> dict:
    """Call cloud model and return the raw assistant message dict."""
    headers, payload, timeout = self._build_openrouter_request(messages, **kwargs)
    if self._tool_schemas:
      payload["tools"] = self._tool_schemas

    request = Request(
      "https://openrouter.ai/api/v1/chat/completions",
      data=json.dumps(payload).encode("utf-8"),
      headers=headers,
      method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
      body = json.load(response)
    self._record_openrouter_usage(payload["model"], body)
    msg = self._openrouter_choice(body)["message"]
    _decode_tool_arguments(msg)
    return msg

  async def _acloud_call_raw(self, messages: list[dict], **kwargs) -> dict:
    """Async cloud call returning raw assistant message dict."""
    headers, payload, timeout = self._build_openrouter_request(messages, **kwargs)
    if self._tool_schemas:
      payload["tools"] = self._tool_schemas

    async with httpx.AsyncClient(timeout=timeout) as client:
      response = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
      )
      response.raise_for_status()
      body = response.json()
    self._record_openrouter_usage(payload["model"], body)
    msg = self._openrouter_choice(body)["message"]
    _decode_tool_arguments(msg)
    return msg

  # --- Tool execution and agentic loop ---

  def _execute_tool_call(self, tool_call: dict) -> str:
    """Dispatch a tool call to its handler, return result as string."""
    fn_name = tool_call["function"]["name"]
    fn_args, error = _tool_arguments(tool_call)
    if error:
      return error
    handler = self._tool_handlers.get(fn_name)
    if handler is None:
      return f"Error: unknown tool '{fn_name}'"
    try:
      result = handler(**fn_args)
      return str(result)
    except Exception as e:
      return f"Error calling {fn_name}: {e}"

  def _run_tool_loop(self, messages: list[dict], **kwargs) -> tuple[str, list[dict]]:
    """Call LLM in a loop, executing tool calls until a final text response."""
    for _ in range(self.max_tool_rounds):
      if self.use_openrouter:
        assistant_msg = self._cloud_call_raw(messages, **kwargs)
      else:
        assistant_msg = self._local_call_raw(messages, **kwargs)

      messages.append(assistant_msg)

      if not assistant_msg.get("tool_calls"):
        return assistant_msg.get("content") or "", messages

      for tc in assistant_msg["tool_calls"]:
        result = self._execute_tool_call(tc)
        messages.append(
          {
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": result,
          }
        )

    raise RuntimeError(f"Tool loop exceeded {self.max_tool_rounds} rounds")

  async def _arun_tool_loop(self, messages: list[dict], **kwargs) -> tuple[str, list[dict]]:
    """Async tool loop. Supports both sync and async tool handlers."""
    import asyncio

    for _ in range(self.max_tool_rounds):
      if self.use_openrouter:
        assistant_msg = await self._acloud_call_raw(messages, **kwargs)
      else:
        assistant_msg = await asyncio.to_thread(self._local_call_raw, messages, **kwargs)

      messages.append(assistant_msg)

      if not assistant_msg.get("tool_calls"):
        return assistant_msg.get("content") or "", messages

      for tc in assistant_msg["tool_calls"]:
        fn_name = tc["function"]["name"]
        fn_args, error = _tool_arguments(tc)
        handler = self._tool_handlers.get(fn_name)
        if error:
          result = error
        elif handler is None:
          result = f"Error: unknown tool '{fn_name}'"
        else:
          try:
            if asyncio.iscoroutinefunction(handler):
              result = str(await handler(**fn_args))
            else:
              result = str(await asyncio.to_thread(handler, **fn_args))
          except Exception as e:
            result = f"Error calling {fn_name}: {e}"
        messages.append(
          {
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": result,
          }
        )

    raise RuntimeError(f"Tool loop exceeded {self.max_tool_rounds} rounds")

  # --- Existing call methods (non-tool path) ---

  def _local_call(self, messages: list[dict], **kwargs) -> str | OutputSchema:
    """Call local Ollama model."""
    response: ChatResponse = ollama_chat(
      model=self.model_name,
      think=self.think,
      messages=messages,
      format=self.output_schema.model_json_schema() if self.output_schema else None,
      **kwargs,
    )
    return self._parse_response(response.message.content)

  def _cloud_call(self, messages: list[dict], **kwargs) -> str | OutputSchema:
    """Call cloud model."""
    return self._openrouter_call(messages, **kwargs)

  async def _acloud_call(self, messages: list[dict], **kwargs) -> str | OutputSchema:
    """Async cloud call."""
    return await self._aopenrouter_call(messages, **kwargs)

  def _build_openrouter_request(
    self, messages: list[dict], **kwargs,
  ) -> tuple[dict, dict, float]:
    """Headers, JSON payload and HTTP timeout (seconds) for one OpenRouter request.

    ``timeout`` is a client setting, never part of the request body: the
    per-call kwarg wins, then ``LLM(timeout=)``, then ``OPENROUTER_HTTP_TIMEOUT``.
    """
    timeout = kwargs.pop("timeout", None) or self.timeout or OPENROUTER_HTTP_TIMEOUT
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
      raise RuntimeError("OPENROUTER_API_KEY is not configured")

    headers = {
      "Authorization": f"Bearer {api_key}",
      "Content-Type": "application/json",
    }
    http_referer = kwargs.pop("http_referer", None) or os.getenv("OPENROUTER_HTTP_REFERER")
    x_title = kwargs.pop("x_title", None) or os.getenv("OPENROUTER_X_TITLE")
    if http_referer:
      headers["HTTP-Referer"] = http_referer
    if x_title:
      headers["X-Title"] = x_title

    provider = dict(kwargs.pop("provider", {}) or {})
    if self.output_schema:
      provider.setdefault("require_parameters", True)

    payload = {
      "model": self.model_name.removeprefix("openrouter/"),
      "messages": messages,
      "usage": {"include": True},
      **kwargs,
    }
    if provider:
      payload["provider"] = provider

    if self.think is True and "reasoning" not in payload:
      payload["reasoning"] = {"effort": "low"}
    if self.output_schema:
      payload["response_format"] = _openrouter_response_format(self.output_schema)
      if not payload.get("stream"):
        plugins = list(payload.get("plugins") or [])
        if not any(
          plugin.get("id") == "response-healing" for plugin in plugins if isinstance(plugin, dict)
        ):
          plugins.append({"id": "response-healing"})
        payload["plugins"] = plugins

    return headers, payload, timeout

  @staticmethod
  def _openrouter_choice(body: dict) -> dict:
    """The first choice of an OpenRouter completion body.

    OpenRouter reports errors that occur while the model generates with HTTP 200
    and an ``error`` object in place of ``choices``.
    """
    choices = body.get("choices")
    if not choices:
      error = body.get("error")
      if error:
        raise LLMResponseError(f"OpenRouter returned an error instead of a completion: {error}")
      raise LLMResponseError(f"OpenRouter response has no choices (keys: {sorted(body)})")
    return choices[0]

  def _parse_openrouter_body(self, body: dict) -> str | OutputSchema:
    choice = self._openrouter_choice(body)
    content = choice["message"].get("content")
    if content is None:
      raise LLMResponseError(
        f"OpenRouter returned no content (finish_reason={choice.get('finish_reason')!r})"
      )
    return self._parse_response(content)

  @staticmethod
  def _record_openrouter_usage(model: str, body: dict) -> None:
    """Meter one OpenRouter response in the cost ledger.

    Called before the body is parsed, so a charged call whose body cannot be
    parsed is still recorded exactly once. ``usage.cost`` is OpenRouter's own
    figure (the request sets ``usage.include``); ``id`` is the generation id.
    """
    from merceka_core import costs as _costs

    usage = body.get("usage") or {}
    _costs.record(
      source="openrouter",
      model=model,
      usage=usage,
      usd=usage.get("cost"),
      request_id=body.get("id"),
    )

  def _openrouter_call(self, messages: list[dict], **kwargs) -> str | OutputSchema:
    headers, payload, timeout = self._build_openrouter_request(messages, **kwargs)
    data = json.dumps(payload).encode("utf-8")

    for attempt in range(_RETRY_MAX_ATTEMPTS):
      request = Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers=headers,
        method="POST",
      )
      try:
        with urlopen(request, timeout=timeout) as response:
          body = json.load(response)
        self._record_openrouter_usage(payload["model"], body)
        return self._parse_openrouter_body(body)
      except urllib.error.HTTPError as exc:
        if exc.code not in _RETRY_STATUS_CODES or attempt == _RETRY_MAX_ATTEMPTS - 1:
          raise
        retry_after = _retry_after_seconds(exc.headers)
        delay = _retry_delay(attempt, retry_after)
        _logger.warning(
          "OpenRouter HTTP %d, retrying in %.2fs (%d/%d)",
          exc.code,
          delay,
          attempt + 1,
          _RETRY_MAX_ATTEMPTS,
        )
        time.sleep(delay)
      except urllib.error.URLError as exc:
        # Retry only failures before any connection existed; see retry.py.
        if not _urlerror_never_sent(exc) or attempt == _RETRY_MAX_ATTEMPTS - 1:
          raise
        delay = _retry_delay(attempt)
        _logger.warning(
          "OpenRouter connection error %s, retrying in %.2fs", type(exc.reason).__name__, delay
        )
        time.sleep(delay)
    # Unreachable (the loop either returns or raises on the last attempt).
    raise RuntimeError("retry loop exhausted without return")

  async def _aopenrouter_call(self, messages: list[dict], **kwargs) -> str | OutputSchema:
    import asyncio

    headers, payload, timeout = self._build_openrouter_request(messages, **kwargs)

    for attempt in range(_RETRY_MAX_ATTEMPTS):
      try:
        async with httpx.AsyncClient(timeout=timeout) as client:
          response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
          )
          response.raise_for_status()
          body = response.json()
        self._record_openrouter_usage(payload["model"], body)
        return self._parse_openrouter_body(body)
      except httpx.HTTPStatusError as exc:
        if (
          exc.response.status_code not in _RETRY_STATUS_CODES or attempt == _RETRY_MAX_ATTEMPTS - 1
        ):
          raise
        retry_after = _retry_after_seconds(exc.response.headers)
        delay = _retry_delay(attempt, retry_after)
        _logger.warning(
          "OpenRouter HTTP %d, retrying in %.2fs (%d/%d)",
          exc.response.status_code,
          delay,
          attempt + 1,
          _RETRY_MAX_ATTEMPTS,
        )
        await asyncio.sleep(delay)
      except _RETRY_HTTPX_ERRORS as exc:
        # Connect/pool failures only: a read or write timeout can follow a
        # request the provider already billed; see retry.py.
        if attempt == _RETRY_MAX_ATTEMPTS - 1:
          raise
        delay = _retry_delay(attempt)
        _logger.warning(
          "OpenRouter connection error %s, retrying in %.2fs", type(exc).__name__, delay
        )
        await asyncio.sleep(delay)
    raise RuntimeError("retry loop exhausted without return")

  async def agenerate(self, message: str, **kwargs) -> str | OutputSchema:
    """Async one-shot generation. Does not maintain conversation history."""
    import asyncio

    try:
      return await self._agenerate_primary(message, **kwargs)
    except _FALLBACK_ERRORS as e:
      # Building a local fallback runs _verify (a blocking Ollama request, or
      # a model pull), so it happens in a worker thread, not on the event loop.
      target = await asyncio.to_thread(self._cascade_target, kwargs)
      if target is None:
        raise
      _logger.warning(
        "Primary LLM failed (%s), falling back to %s", type(e).__name__, self.fallback
      )
      fb, fb_kwargs = target
      return await fb.agenerate(message, **fb_kwargs)

  async def _agenerate_primary(self, message: str, **kwargs) -> str | OutputSchema:
    """Async primary generation dispatch. Mirrors _generate_primary exactly."""
    import asyncio

    messages = [create_message(self.system_prompt, "system"), create_message(message, "user")]
    backend = self._select_backend()
    if backend == _BACKEND_TOOLS_FALLBACK:
      fb, fb_kwargs = await asyncio.to_thread(self._tools_fallback_call, kwargs)
      return await fb._agenerate_primary(message, **fb_kwargs)
    if backend == _BACKEND_CLAUDE:
      return await asyncio.to_thread(self._claude_call, message, **kwargs)
    if backend == _BACKEND_CODEX:
      return await asyncio.to_thread(self._codex_call, message, **kwargs)
    if backend == _BACKEND_TOOL_LOOP:
      text, _ = await self._arun_tool_loop(messages, **kwargs)
      return text
    if backend == _BACKEND_OPENROUTER:
      return await self._acloud_call(messages, **kwargs)
    return await asyncio.to_thread(self._local_call, messages, **kwargs)

  def _parse_response(self, content) -> str | OutputSchema:
    """Parse raw response content, validating against schema if set."""
    if content is None:
      raise LLMResponseError(f"{self.model_name} returned no content")
    if self.output_schema:
      if isinstance(content, str):
        return self.output_schema.model_validate_json(content)
      return self.output_schema.model_validate(content)
    if isinstance(content, str):
      return content
    return json.dumps(content, ensure_ascii=False)

  def _response_to_history_content(self, response: BaseModel) -> str:
    content = getattr(response, "content", None)
    if isinstance(content, str) and content:
      return content
    return response.model_dump_json()

  async def agenerate_batch(
    self,
    messages: list[str],
    concurrency: int = 10,
    show_progress: bool = True,
    *,
    return_exceptions: bool = False,
    **kwargs,
  ) -> list[str | OutputSchema]:
    """Batch async generation with concurrency control.

    Items start in input order. When one fails, queued items never start and
    the calls already in flight finish (their spend is metered), then the first
    failure is raised. With ``return_exceptions=True`` every item runs and a
    failed item's slot holds its exception instead.

    Args:
        messages: List of input messages to process
        concurrency: Max parallel requests (default 10)
        show_progress: Show tqdm progress bar
        return_exceptions: Return failures in place instead of raising
        **kwargs: Additional args passed to the API (e.g., temperature)

    Returns:
        List of responses in same order as inputs
    """
    import asyncio

    semaphore = asyncio.Semaphore(concurrency)
    failures: list[Exception] = []
    progress = tqdm(total=len(messages), desc="Processing") if show_progress else None

    async def process_one(message: str):
      async with semaphore:
        if failures:
          return None  # a call already failed: start no new paid call
        try:
          return await self.agenerate(message, **kwargs)
        except Exception as exc:
          if not return_exceptions:
            failures.append(exc)
          return exc
        finally:
          if progress is not None:
            progress.update(1)

    try:
      results = await asyncio.gather(*(process_one(msg) for msg in messages))
    finally:
      if progress is not None:
        progress.close()
    if failures:
      raise failures[0]
    # No slot is None (items are skipped only after a failure, which raises).
    # Exceptions appear only with return_exceptions=True, as documented.
    return cast(list[str | OutputSchema], results)

  def _resolve_timeout(self, kwargs: dict) -> int:
    """Resolve the subprocess timeout: per-call kwarg > instance default > module default."""
    if "timeout" in kwargs:
      return kwargs["timeout"]
    if self.timeout is not None:
      return self.timeout
    return CLAUDE_CLI_TIMEOUT

  @contextlib.contextmanager
  def _claude_workdir(self):
    """Working directory for Claude CLI runs.

    Read/Grep/Glob run without approval inside the working directory, so it must
    be a declared directory, never the caller's cwd (which may hold a .env). The
    first ``add_dirs`` entry is used when present, otherwise a scratch directory.
    """
    if self.add_dirs and Path(self.add_dirs[0]).is_dir():
      yield str(self.add_dirs[0])
      return
    with tempfile.TemporaryDirectory(prefix="merceka-claude-") as scratch:
      yield scratch

  def _claude_call(self, message: str, **kwargs) -> str | OutputSchema:
    """Call Claude CLI via subprocess.

    Supports Claude Code native tool calling via --allowedTools and
    --add-dir flags. When these are set, Claude Code handles file
    access (Read, Grep, Glob) internally — no Python tool loop needed.
    """
    cmd = _cli.claude_command(
      self.model_name.removeprefix("claude/"),
      system_prompt=self.system_prompt,
      add_dirs=self.add_dirs,
      allowed_tools=self.allowed_tools,
    )
    timeout = self._resolve_timeout(kwargs)
    env = _cli.claude_env()

    with self._claude_workdir() as cwd:
      result = subprocess.run(
        cmd,
        input=message,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=cwd,
      )
    if result.returncode != 0:
      raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    content = result.stdout.strip()
    return self._parse_response(content)

  def _codex_call(self, message: str, **kwargs) -> str | OutputSchema:
    """Call OpenAI Codex CLI via subprocess (`codex exec`).

    Runs on the Codex subscription (ChatGPT auth), not API billing.
    Supports vision via kwargs["images"] (list of file paths, passed as
    -i flags). The model alias after "codex/" is passed as -m; use
    "codex/default" to use the user's configured default model.
    No structured-output support — ask for JSON in the prompt and parse
    the response yourself (same approach as the Claude CLI provider).
    """
    cmd = _cli.codex_exec_command(
      self.model_name.removeprefix("codex/"),
      ephemeral=True,
      images=kwargs.get("images", []) or [],
    )

    # codex exec has no --system-prompt flag; prepend it to the message
    prompt = f"{self.system_prompt}\n\n{message}" if self.system_prompt else message

    timeout = self._resolve_timeout(kwargs)
    result = subprocess.run(
      cmd,
      input=prompt,
      capture_output=True,
      text=True,
      timeout=timeout,
      env=_cli.codex_env(),
    )
    if result.returncode != 0:
      raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    # codex exec writes log/session lines to stderr; stdout is the final message
    return self._parse_response(result.stdout.strip())

  def _claude_stream(self, message: str, **kwargs):
    """Stream tokens from Claude CLI via Popen + stream-json.

    Yields text chunks as Claude generates them. Tool use happens
    internally (Claude Code handles Read/Grep/Glob) — only text
    deltas are yielded.
    """
    cmd = _cli.claude_command(
      self.model_name.removeprefix("claude/"),
      system_prompt=self.system_prompt,
      add_dirs=self.add_dirs,
      allowed_tools=self.allowed_tools,
      stream=True,
    )
    with self._claude_workdir() as cwd:
      env = _cli.claude_env()
      process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        cwd=cwd,
      )
      finished = False  # stdout was read to a result event or to EOF
      result_seen = False
      result_error: str | None = None
      stderr_tail = ""
      try:
        try:
          # Send message and close stdin so Claude starts processing.
          process.stdin.write(message)
          process.stdin.close()
        except BrokenPipeError:
          pass  # the CLI exited at once; its exit status and stderr say why

        for line in process.stdout:
          line = line.strip()
          if not line:
            continue
          try:
            obj = json.loads(line)
          except json.JSONDecodeError:
            continue

          text = _cli.claude_stream_text_delta(obj)
          if text is not None:
            yield text
          elif _cli.is_claude_result_event(obj):
            result_seen = True
            result_error = _claude_result_error(obj)
            break
        finished = True
      finally:
        # Runs on normal completion, on errors, and when the consumer abandons
        # the stream (GeneratorExit): the child is always reaped, never waited
        # on without a bound.
        signalled = _stop_stream_process(process, finished=finished)
        # Success is a clean exit, or a clean result event from a child that
        # then had to be signalled because it lingered.
        succeeded = result_error is None and (
          result_seen if signalled else process.returncode == 0
        )
        if finished and not succeeded and process.poll() is not None:
          stderr_tail = _read_pipe_tail(process.stderr, _STREAM_STDERR_TAIL)
        for pipe in (process.stdin, process.stdout, process.stderr):
          with contextlib.suppress(OSError):
            pipe.close()

    if succeeded:
      return
    # The CLI reports failures (usage limit, auth, bad flags) through an error
    # result event and/or a non-zero exit. Raise, so stream_generate's fallback
    # runs instead of an empty answer passing as success.
    detail = "\n".join(part for part in (result_error, stderr_tail.strip()) if part)
    code = process.returncode if process.returncode and not signalled else 1
    raise subprocess.CalledProcessError(code, cmd, stderr=detail or None)

  def stream_generate(self, message: str, **kwargs):
    """Stream tokens from the primary model. Sync generator.

    Claude CLI models stream token deltas. Other models have no token stream:
    the primary's full response is yielded as one chunk, and the fallback
    answers only when the primary fails (``generate``'s own cascade).
    """
    if not self.use_claude:
      yield self.generate(message, **kwargs)
      return

    streamed = False
    try:
      with contextlib.closing(self._claude_stream(message, **kwargs)) as stream:
        for chunk in stream:
          streamed = True
          yield chunk
      return
    except (FileNotFoundError, OSError, subprocess.CalledProcessError) as e:
      # Once chunks reached the consumer, a fallback answer would be appended
      # to a partial one, so the failure is raised instead. The stream ignores
      # Python tools, so this is a primary failure even with tools set.
      target = self._fallback_call(kwargs) if self.fallback and not streamed else None
      if target is None:
        raise
      _logger.warning("Claude stream failed (%s), falling back", type(e).__name__)

    # The fallback answers as one chunk.
    fb, fb_kwargs = target
    yield fb.generate(message, **fb_kwargs)

  async def astream_generate(self, message: str, **kwargs):
    """Async streaming generator. Runs the sync stream in a worker thread."""
    import asyncio
    import threading

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    sentinel = object()
    stop = threading.Event()  # set when the consumer abandons early

    def _put(item):
      try:
        loop.call_soon_threadsafe(q.put_nowait, item)
      except RuntimeError:
        pass  # loop closed during teardown; nothing left to deliver to

    def _run():
      try:
        for chunk in self.stream_generate(message, **kwargs):
          if stop.is_set():
            return
          _put(chunk)
      except Exception as e:
        _put(e)
      finally:
        _put(sentinel)

    producer = loop.run_in_executor(None, _run)
    try:
      while True:
        item = await q.get()
        if item is sentinel:
          break
        if isinstance(item, Exception):
          raise item
        yield item
    finally:
      stop.set()  # bounds the finalizer wait to at most one in-flight chunk
      await asyncio.shield(producer)

  def generate_with_video(
    self,
    message: str,
    video_paths,  # Path | str | list[Path | str]
    *,
    timeout_s: float = 300.0,
    poll_interval_s: float = 5.0,
    **kwargs,
  ) -> str | OutputSchema:
    """One-shot long-context video generation (Gemini only).

    Uploads each video via the Files API, polls until ``state ==
    'ACTIVE'`` (or raises :class:`VideoUploadError` on timeout/
    failure), then calls ``generate_content``. Deletes uploaded
    files afterwards as hygiene.

    The model is set at ``LLM`` construction time. The recommended
    default for video is ``LLM("gemini/gemini-flash-latest")`` — the
    ``gemini-flash-latest`` alias resolves to the newest full-fat Flash
    (currently ``gemini-3-flash-preview``, Dec 2025) and auto-upgrades
    as new Flash models ship. Use ``LLM("gemini/gemini-pro-latest")``
    when you need the Pro tier's deeper reasoning on a curated clip and
    are willing to pay for it.

    Args:
      message: The text prompt that accompanies the video(s).
      video_paths: A single path or a list of paths. Always
        normalized to list internally so a future multi-clip
        signature does not require a public-API break.
      timeout_s: How long to wait for upload to reach ACTIVE.
      poll_interval_s: Seconds between ``files.get`` polls.

    Raises:
      VideoUploadError: File FAILED or exceeded ``timeout_s``, or the request
        was rejected (400/401/403/404). Uploaded files are deleted either way.
      VideoNotFoundError: Path does not exist on disk.
      VideoBackendError: 5xx / transient inference failure.
    """
    if not self.use_gemini:
      raise ValueError(
        "generate_with_video requires a Gemini model (model_name must start with 'gemini/')."
      )
    return _gemini_video_call(
      self, message, video_paths, timeout_s=timeout_s, poll_interval_s=poll_interval_s, **kwargs
    )

  async def agenerate_with_video(
    self,
    message: str,
    video_paths,
    *,
    timeout_s: float = 300.0,
    poll_interval_s: float = 5.0,
    **kwargs,
  ) -> str | OutputSchema:
    """Async mirror of :meth:`generate_with_video`."""
    import asyncio

    if not self.use_gemini:
      raise ValueError(
        "agenerate_with_video requires a Gemini model (model_name must start with 'gemini/')."
      )
    return await asyncio.to_thread(
      _gemini_video_call,
      self,
      message,
      video_paths,
      timeout_s=timeout_s,
      poll_interval_s=poll_interval_s,
      **kwargs,
    )

  def _verify(self):
    """Verify the model is available, download if missing.

    Ollama lists installed models with their tag, and an untagged name means
    ``:latest``, so ``gemma3`` is installed when ``gemma3:latest`` is listed.
    """
    if _with_default_tag(self.model_name) not in {
      _with_default_tag(name) for name in list_local_models()
    }:
      _download_model(self.model_name)


def _with_default_tag(model_name: str) -> str:
  """``model_name`` with Ollama's implicit ``:latest`` tag made explicit.

  Only a colon in the last path segment is a tag (``host:port/model`` is not).
  """
  return model_name if ":" in model_name.rsplit("/", 1)[-1] else f"{model_name}:latest"


# Gemini surface moved to merceka_core.llm_gemini; re-exported for back-compat.
from merceka_core.llm_gemini import (  # noqa: E402, F401 — re-exported for back-compat
  _build_video_config,
  _gemini_image_call,
  _extract_grounding,
  _gemini_client,
  _gemini_poll_until_active,
  _gemini_video_call,
  _generate_with_search_grounding_sync,
  generate_with_search_grounding,
)
