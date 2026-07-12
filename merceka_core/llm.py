"""Load and run ollama models"""

__all__ = ['Tool', 'list_local_models', 'create_message', 'create_message_with_resource', 'create_ollama_vision_message',
           'tool_from_callable', 'OutputSchema', 'LLM', 'generate_with_search_grounding']

import dotenv
import json
import logging
import os
import subprocess
import httpx
import time
import urllib.error

_logger = logging.getLogger(__name__)

CLAUDE_CLI_TIMEOUT = 120  # seconds


dotenv.load_dotenv()


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


from typing import Callable


def _chat_one(model: str, message: str, think: Optional[bool] = None, **kwargs):
  """Chat with the model."""
  return ollama_chat(
    model=model, think=think, messages=[create_message(message)], **kwargs
  ).message.content


from pydantic import BaseModel


from ollama import ChatResponse
import litellm
litellm.suppress_debug_info = True  # Stop printing "Provider List" spam
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
  _RETRY_MAX_ATTEMPTS,
  _RETRY_MAX_DELAY,
  _RETRY_STATUS_CODES,
  _retry_delay,
  _retry_after_seconds,
)

class LLM:
  """A class for interacting with an LLM."""

  def __init__(
    self,
    model_name: str,  # The name of the model to use
    system_prompt: str = "",  # The system prompt to use.
    think: Optional[bool] = None,  # Whether to enable thinking mode
    output_schema: Optional[
      type[BaseModel]
    ] = None,  # Schema for structured output
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
    self.use_openrouter = (not self.use_claude and not self.use_codex
                           and not self.use_gemini and "openrouter" in model_name)
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

    if (not self.use_openrouter and not self.use_claude and not self.use_codex
        and not self.use_gemini):
      self._verify()

  def _fallback_llm(self, model_name: Optional[str] = None) -> "LLM":
    """Construct a fallback LLM preserving the full configuration of this one."""
    return LLM(model_name or self.fallback, system_prompt=self.system_prompt,
               think=self.think, output_schema=self.output_schema,
               tools=self._original_tools, max_tool_rounds=self.max_tool_rounds,
               add_dirs=self.add_dirs, allowed_tools=self.allowed_tools,
               timeout=self.timeout)

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
        "or route text through an openrouter/ model.")
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
        "tool-capable model, or drop tools=.")
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
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError,
            FileNotFoundError, ConnectionError, OSError,
            httpx.HTTPError, urllib.error.URLError,
            VideoBackendError) as e:
      if self.fallback:
        _logger.warning("Primary LLM failed (%s), falling back to %s", type(e).__name__, self.fallback)
        return self._fallback_llm().generate(message, **kwargs)
      raise

  def _generate_primary(self, message: str, **kwargs) -> str | OutputSchema:
    """Primary generation dispatch."""
    messages = [create_message(self.system_prompt, "system"), create_message(message, "user")]
    backend = self._select_backend()
    if backend == _BACKEND_TOOLS_FALLBACK:
      _logger.info("%s can't run Python tool callables, using fallback %s",
                   self.model_name, self.fallback)
      return self._fallback_llm().generate(message, **kwargs)
    if backend == _BACKEND_CLAUDE:
      return self._claude_call(message, **kwargs)
    if backend == _BACKEND_CODEX:
      return self._codex_call(message, **kwargs)
    if backend == _BACKEND_TOOL_LOOP:
      text, _ = self._run_tool_loop(messages, **kwargs)
      return text
    if backend == _BACKEND_OPENROUTER:
      return self._cloud_call(messages, **kwargs)
    return self._local_call(messages, **kwargs)

  def chat(self, message: str, **kwargs) -> str | OutputSchema:
    """Multi-turn chat. Maintains conversation history."""
    if self.use_gemini:
      self._select_backend()  # raises with the Gemini guidance message
    self.messages.append(create_message(message, "user"))

    if self._tool_schemas:
      text, self.messages = self._run_tool_loop(list(self.messages), **kwargs)
      return text

    if self.use_claude:
      # Claude CLI is one-shot; send full history as context (exclude system, it's in --system-prompt)
      history = "\n".join(f"{m['role']}: {m['content']}" for m in self.messages if m.get('content') and m['role'] != 'system')
      response = self._claude_call(history, **kwargs)
    elif self.use_openrouter:
      response = self._cloud_call(self.messages, **kwargs)
    else:
      response = self._local_call(self.messages, **kwargs)

    # Extract content for history: schema responses have .content, plain responses are strings
    if isinstance(response, BaseModel):
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

    Supports OpenRouter cloud models and local Ollama vision models. Claude
    CLI is not supported (the CLI takes stdin text only). Does not maintain
    conversation history.

    Args:
      message: The text prompt to accompany the resource.
      resource_path: Path to the file (image or PDF).
      **kwargs: Additional args passed to the API.

    Returns:
      Model response as string or OutputSchema.
    """
    if self.use_claude:
      raise ValueError(
        "generate_with_resource is not supported for Claude CLI models — "
        "the CLI accepts stdin text only. Use an openrouter, gemini, or ollama model."
      )

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

    Mirrors :meth:`generate_with_resource` but runs the local Ollama call in a
    worker thread so it doesn't block the event loop. Supports OpenRouter and
    local Ollama vision models; Claude CLI is not supported.
    """
    import asyncio

    if self.use_claude:
      raise ValueError(
        "agenerate_with_resource is not supported for Claude CLI models — "
        "the CLI accepts stdin text only. Use an openrouter, gemini, or ollama model."
      )

    if self.use_gemini:
      return await asyncio.to_thread(
        _gemini_image_call, self, message, resource_path, **kwargs
      )

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
        tool_calls.append({
          "id": f"call_{i}",
          "type": "function",
          "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
          },
        })
    return {
      "role": "assistant",
      "content": msg.content,
      "tool_calls": tool_calls,
    }

  def _cloud_call_raw(self, messages: list[dict], **kwargs) -> dict:
    """Call cloud model and return the raw assistant message dict."""
    headers, payload = self._build_openrouter_request(messages, **kwargs)
    if self._tool_schemas:
      payload["tools"] = self._tool_schemas

    request = Request(
      "https://openrouter.ai/api/v1/chat/completions",
      data=json.dumps(payload).encode("utf-8"),
      headers=headers,
      method="POST",
    )
    with urlopen(request, timeout=120) as response:
      body = json.load(response)
    msg = body["choices"][0]["message"]
    # Normalize: ensure arguments is a dict
    if msg.get("tool_calls"):
      for tc in msg["tool_calls"]:
        args = tc["function"].get("arguments")
        if isinstance(args, str):
          tc["function"]["arguments"] = json.loads(args)
    return msg

  async def _acloud_call_raw(self, messages: list[dict], **kwargs) -> dict:
    """Async cloud call returning raw assistant message dict."""
    headers, payload = self._build_openrouter_request(messages, **kwargs)
    if self._tool_schemas:
      payload["tools"] = self._tool_schemas

    async with httpx.AsyncClient(timeout=120.0) as client:
      response = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
      )
      response.raise_for_status()
      body = response.json()
    msg = body["choices"][0]["message"]
    if msg.get("tool_calls"):
      for tc in msg["tool_calls"]:
        args = tc["function"].get("arguments")
        if isinstance(args, str):
          tc["function"]["arguments"] = json.loads(args)
    return msg

  # --- Tool execution and agentic loop ---

  def _execute_tool_call(self, tool_call: dict) -> str:
    """Dispatch a tool call to its handler, return result as string."""
    fn_name = tool_call["function"]["name"]
    fn_args = tool_call["function"]["arguments"]
    if isinstance(fn_args, str):
      fn_args = json.loads(fn_args)
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
        messages.append({
          "role": "tool",
          "tool_call_id": tc["id"],
          "content": result,
        })

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
        fn_args = tc["function"]["arguments"]
        if isinstance(fn_args, str):
          fn_args = json.loads(fn_args)
        handler = self._tool_handlers.get(fn_name)
        if handler is None:
          result = f"Error: unknown tool '{fn_name}'"
        else:
          try:
            if asyncio.iscoroutinefunction(handler):
              result = str(await handler(**fn_args))
            else:
              result = str(await asyncio.to_thread(handler, **fn_args))
          except Exception as e:
            result = f"Error calling {fn_name}: {e}"
        messages.append({
          "role": "tool",
          "tool_call_id": tc["id"],
          "content": result,
        })

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

  def _build_openrouter_request(self, messages: list[dict], **kwargs) -> tuple[dict, dict]:
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
        if not any(plugin.get("id") == "response-healing" for plugin in plugins if isinstance(plugin, dict)):
          plugins.append({"id": "response-healing"})
        payload["plugins"] = plugins

    return headers, payload

  def _parse_openrouter_body(self, body: dict) -> str | OutputSchema:
    return self._parse_response(body["choices"][0]["message"]["content"])

  def _openrouter_call(self, messages: list[dict], **kwargs) -> str | OutputSchema:
    headers, payload = self._build_openrouter_request(messages, **kwargs)
    data = json.dumps(payload).encode("utf-8")

    for attempt in range(_RETRY_MAX_ATTEMPTS):
      request = Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers=headers,
        method="POST",
      )
      try:
        with urlopen(request, timeout=120) as response:
          body = json.load(response)
        return self._parse_openrouter_body(body)
      except urllib.error.HTTPError as exc:
        if exc.code not in _RETRY_STATUS_CODES or attempt == _RETRY_MAX_ATTEMPTS - 1:
          raise
        retry_after = _retry_after_seconds(exc.headers)
        delay = _retry_delay(attempt, retry_after)
        _logger.warning("OpenRouter HTTP %d, retrying in %.2fs (%d/%d)", exc.code, delay, attempt + 1, _RETRY_MAX_ATTEMPTS)
        time.sleep(delay)
      except (ConnectionRefusedError, ConnectionResetError, urllib.error.URLError) as exc:
        if attempt == _RETRY_MAX_ATTEMPTS - 1:
          raise
        delay = _retry_delay(attempt)
        _logger.warning("OpenRouter connection error %s, retrying in %.2fs", type(exc).__name__, delay)
        time.sleep(delay)
    # Unreachable (the loop either returns or raises on the last attempt).
    raise RuntimeError("retry loop exhausted without return")

  async def _aopenrouter_call(self, messages: list[dict], **kwargs) -> str | OutputSchema:
    import asyncio

    headers, payload = self._build_openrouter_request(messages, **kwargs)

    for attempt in range(_RETRY_MAX_ATTEMPTS):
      try:
        async with httpx.AsyncClient(timeout=120.0) as client:
          response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
          )
          response.raise_for_status()
          body = response.json()
        return self._parse_openrouter_body(body)
      except httpx.HTTPStatusError as exc:
        if exc.response.status_code not in _RETRY_STATUS_CODES or attempt == _RETRY_MAX_ATTEMPTS - 1:
          raise
        retry_after = _retry_after_seconds(exc.response.headers)
        delay = _retry_delay(attempt, retry_after)
        _logger.warning("OpenRouter HTTP %d, retrying in %.2fs (%d/%d)", exc.response.status_code, delay, attempt + 1, _RETRY_MAX_ATTEMPTS)
        await asyncio.sleep(delay)
      except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout, ConnectionRefusedError, ConnectionResetError) as exc:
        if attempt == _RETRY_MAX_ATTEMPTS - 1:
          raise
        delay = _retry_delay(attempt)
        _logger.warning("OpenRouter connection error %s, retrying in %.2fs", type(exc).__name__, delay)
        await asyncio.sleep(delay)
    raise RuntimeError("retry loop exhausted without return")

  async def agenerate(self, message: str, **kwargs) -> str | OutputSchema:
    """Async one-shot generation. Does not maintain conversation history."""
    try:
      return await self._agenerate_primary(message, **kwargs)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError,
            FileNotFoundError, ConnectionError, OSError,
            httpx.HTTPError, urllib.error.URLError,
            VideoBackendError) as e:
      if self.fallback:
        _logger.warning("Primary LLM failed (%s), falling back to %s", type(e).__name__, self.fallback)
        return await self._fallback_llm().agenerate(message, **kwargs)
      raise

  async def _agenerate_primary(self, message: str, **kwargs) -> str | OutputSchema:
    """Async primary generation dispatch. Mirrors _generate_primary exactly."""
    import asyncio

    messages = [create_message(self.system_prompt, "system"), create_message(message, "user")]
    backend = self._select_backend()
    if backend == _BACKEND_TOOLS_FALLBACK:
      _logger.info("%s can't run Python tool callables, using fallback %s",
                   self.model_name, self.fallback)
      return await self._fallback_llm().agenerate(message, **kwargs)
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
    assert content is not None, "No content was returned"
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
    **kwargs,
  ) -> list[str | OutputSchema]:
    """Batch async generation with concurrency control.

    Args:
        messages: List of input messages to process
        concurrency: Max parallel requests (default 10)
        show_progress: Show tqdm progress bar
        **kwargs: Additional args passed to the API (e.g., temperature)

    Returns:
        List of responses in same order as inputs
    """
    import asyncio
    from tqdm.asyncio import tqdm_asyncio

    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(message: str) -> str | OutputSchema:
      async with semaphore:
        return await self.agenerate(message, **kwargs)

    tasks = [process_one(msg) for msg in messages]
    if show_progress:
      return await tqdm_asyncio.gather(*tasks, desc="Processing")
    return await asyncio.gather(*tasks)

  def _resolve_timeout(self, kwargs: dict) -> int:
    """Resolve the subprocess timeout: per-call kwarg > instance default > module default."""
    if "timeout" in kwargs:
      return kwargs["timeout"]
    if self.timeout is not None:
      return self.timeout
    return CLAUDE_CLI_TIMEOUT

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

    result = subprocess.run(
      cmd,
      input=message,
      capture_output=True,
      text=True,
      timeout=timeout,
      env=env,
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
    env = _cli.claude_env()
    process = subprocess.Popen(
      cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
      stderr=subprocess.PIPE, text=True, bufsize=1, env=env,
    )
    # Send message and close stdin so Claude starts processing
    process.stdin.write(message)
    process.stdin.close()

    try:
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
          break
    finally:
      process.stdout.close()
      process.stderr.close()
      process.wait()

  def stream_generate(self, message: str, **kwargs):
    """Stream tokens from the primary model. Sync generator.

    Falls back to yielding the full response as one chunk for
    non-Claude models.
    """
    if self.use_claude:
      try:
        yield from self._claude_stream(message, **kwargs)
        return
      except (FileNotFoundError, OSError) as e:
        if self.fallback:
          _logger.warning("Claude stream failed (%s), falling back", type(e).__name__)
        else:
          raise

    # Fallback: generate full response and yield as one chunk
    fb = self._fallback_llm(self.fallback or self.model_name)
    yield fb.generate(message, **kwargs)

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
      VideoUploadError: File FAILED or exceeded ``timeout_s``.
      VideoNotFoundError: Path does not exist on disk.
      VideoBackendError: 5xx / transient inference failure.
    """
    if not self.use_gemini:
      raise ValueError(
        "generate_with_video requires a Gemini model (model_name must "
        "start with 'gemini/')."
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
        "agenerate_with_video requires a Gemini model (model_name must "
        "start with 'gemini/')."
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
    """Verify the model is available, download if missing."""
    if self.model_name not in list_local_models():
      _download_model(self.model_name)

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
