"""Message construction, tool schemas, and structured-output helpers.

Pure builders shared by the LLM class and downstream callers. No transport
logic lives here.
"""

from __future__ import annotations

import base64
import inspect
import mimetypes
import re as _re
from pathlib import Path
from typing import Callable, Literal, Optional, get_type_hints

from pydantic import BaseModel

Tool = Callable | tuple[dict, Callable]  # auto-schema or (raw_schema, handler)


def create_message(content: Optional[str], role: Literal["user", "assistant", "system"] = "user"):
  """Create a message for the chat."""
  return {"role": role, "content": content}


def create_message_with_resource(
  text: str,
  resource_path: Path | str,
  role: Literal["user", "assistant"] = "user",
) -> dict:
  """Create a message with an attached file (image/PDF) for vision models.
  
  Args:
    text: The text prompt to accompany the resource
    resource_path: Path to the file (image or PDF)
    role: Message role (user or assistant)
    
  Returns:
    Message dict in litellm vision format with base64-encoded content
  """
  resource_path = Path(resource_path)
  
  # Read and encode file
  file_bytes = resource_path.read_bytes()
  base64_data = base64.b64encode(file_bytes).decode("utf-8")
  
  # Detect MIME type
  mime_type, _ = mimetypes.guess_type(str(resource_path))
  if mime_type is None:
    # Default based on extension
    ext = resource_path.suffix.lower()
    mime_map = {
      ".pdf": "application/pdf",
      ".png": "image/png",
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
      ".gif": "image/gif",
      ".webp": "image/webp",
      # Video formats (Gemini long-context video).
      ".mp4": "video/mp4",
      ".mov": "video/quicktime",
      ".webm": "video/webm",
      ".mpeg": "video/mpeg",
      ".mpg": "video/mpeg",
      ".avi": "video/x-msvideo",
      ".flv": "video/x-flv",
      ".wmv": "video/x-ms-wmv",
      ".3gp": "video/3gpp",
    }
    mime_type = mime_map.get(ext, "application/octet-stream")
  
  # Create message with multimodal content
  return {
    "role": role,
    "content": [
      {"type": "text", "text": text},
      {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
      },
    ],
  }


def create_ollama_vision_message(
  text: str,
  image_path: Path | str,
  role: Literal["user", "assistant"] = "user",
) -> dict:
  """Create a vision message in Ollama-native format.

  Ollama-python's ``chat`` expects images as a top-level ``images`` field on the
  message, not embedded in ``content``. The value may be a list of file paths,
  bytes, or base64 strings. We pass a path — Ollama reads and encodes it.

  Args:
    text: The text prompt to accompany the image.
    image_path: Path to the image file on disk.
    role: Message role (user or assistant).

  Returns:
    Message dict in Ollama's native vision format:
    ``{"role": role, "content": text, "images": [str(path)]}``.
  """
  return {
    "role": role,
    "content": text,
    "images": [str(Path(image_path))],
  }


def _python_type_to_json(hint) -> str:
  """Map a Python type annotation to a JSON Schema type string."""
  _TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
  }
  origin = getattr(hint, "__origin__", None)
  if origin is list:
    return "array"
  return _TYPE_MAP.get(hint, "string")


def _parse_param_docs(docstring: str | None) -> dict[str, str]:
  """Extract parameter descriptions from a Google-style Args: section."""
  if not docstring:
    return {}
  result: dict[str, str] = {}
  in_args = False
  current_param: str | None = None
  current_desc_lines: list[str] = []

  for line in docstring.splitlines():
    stripped = line.strip()
    if stripped.lower().startswith("args:"):
      in_args = True
      continue
    if in_args:
      if stripped and not stripped[0].isspace() and not stripped.startswith("-") and ":" not in stripped:
        # Left-aligned non-param line → we've exited the Args section
        if not stripped[0].isalpha() or stripped.endswith(":"):
          break
      # Check for "param_name: description" or "param_name (type): description"
      m = _re.match(r"^(\w+)\s*(?:\([^)]*\))?\s*[:–—-]\s*(.+)", stripped)
      if m:
        if current_param:
          result[current_param] = " ".join(current_desc_lines).strip()
        current_param = m.group(1)
        current_desc_lines = [m.group(2)]
      elif current_param and stripped:
        current_desc_lines.append(stripped)
      elif not stripped and current_param:
        result[current_param] = " ".join(current_desc_lines).strip()
        current_param = None
        current_desc_lines = []

  if current_param:
    result[current_param] = " ".join(current_desc_lines).strip()
  return result

def tool_from_callable(fn: Callable) -> dict:
  """Convert a typed Python function to an OpenAI tool schema dict.

  Extracts function name, docstring, type hints, and default values to produce
  the schema expected by OpenAI/OpenRouter tool calling APIs.

  Args:
    fn: A Python function with type annotations and optional docstring.

  Returns:
    A dict in OpenAI tool format: {"type": "function", "function": {...}}
  """
  sig = inspect.signature(fn)
  try:
    hints = get_type_hints(fn)
  except Exception:
    hints = {}

  doc = inspect.getdoc(fn) or ""
  description = doc.split("\n\n")[0].strip() if doc else fn.__name__
  param_docs = _parse_param_docs(doc)

  properties: dict[str, dict] = {}
  required: list[str] = []

  for name, param in sig.parameters.items():
    hint = hints.get(name)
    prop: dict = {"type": _python_type_to_json(hint) if hint else "string"}
    if name in param_docs:
      prop["description"] = param_docs[name]
    properties[name] = prop
    if param.default is inspect.Parameter.empty:
      required.append(name)

  schema: dict = {
    "type": "function",
    "function": {
      "name": fn.__name__,
      "description": description,
      "parameters": {
        "type": "object",
        "properties": properties,
      },
    },
  }
  if required:
    schema["function"]["parameters"]["required"] = required
  return schema


class OutputSchema(BaseModel):
  """Base class for structured LLM outputs. Subclass this to define your schema.

  The optional `content` field will be used for chat history when present."""

  content: str | None = None


def _schema_name(schema: type[BaseModel]) -> str:
  name = getattr(schema, "__name__", "structured_response")
  return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name) or "structured_response"


def _openrouter_response_format(schema: type[BaseModel]) -> dict:
  return {
    "type": "json_schema",
    "json_schema": {
      "name": _schema_name(schema),
      "strict": True,
      "schema": schema.model_json_schema(),
    },
  }

