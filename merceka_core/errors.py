"""Exception hierarchy for merceka_core.

These are deliberately thin so downstream consumers (slab, videototext,
mindweaver) can distinguish retryable from terminal failures without
importing SDK-specific exception types from google-genai, httpx, ollama,
or the Anthropic SDK.

``VideoNotFoundError`` inherits ``FileNotFoundError`` and
``GpuLockTimeout`` inherits ``TimeoutError`` so the existing
fallback/retry layers in :mod:`merceka_core.llm` keep working without
having to register the new types explicitly.
"""


class VideoUploadError(Exception):
  """Terminal Gemini failure: retrying the same request cannot succeed.

  Raised when a video is rejected at upload time (FAILED processing, never
  reaching ACTIVE, codec/size/quota), and when ``generate_content`` rejects the
  request itself with 400/401/403/404 (bad argument, bad or unauthorised key,
  unknown model). The caller should surface this rather than retry. Does not
  participate in the ``LLM.generate`` fallback cascade.
  """


class VideoBackendError(Exception):
  """Transient Gemini backend failure during inference.

  Raised for 5xx / 429 / connection failures that persist through the shared
  retry policy, and for other unclassified SDK errors, while the model is
  generating. Participates in the fallback cascade.
  """


class VideoNotFoundError(FileNotFoundError):
  """Video path does not exist on disk."""


class LLMResponseError(Exception):
  """The provider answered without a usable completion.

  Raised when an OpenRouter body carries an ``error`` object instead of
  ``choices`` (OpenRouter reports errors that occur while the model generates
  with HTTP 200), and when a response has no content (for example, reasoning
  used up ``max_tokens``). Never retried, because the provider processed and may
  have billed the request. Participates in the ``LLM.generate`` fallback cascade.
  """


class GpuLockTimeout(TimeoutError):
  """Timed out waiting to acquire the cross-process GPU file lock."""
