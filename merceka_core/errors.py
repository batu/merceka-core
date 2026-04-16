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
  """Video rejected by the backend at upload time.

  Raised for codec/size/quota failures that are NOT transient — the
  caller should surface this to the user rather than retry. Does not
  participate in the ``LLM.generate`` fallback cascade.
  """


class VideoBackendError(Exception):
  """Transient backend failure during video inference.

  Raised for 5xx / 429 / timeout failures that happen AFTER a successful
  upload, while the model is generating. Participates in the fallback
  cascade.
  """


class VideoNotFoundError(FileNotFoundError):
  """Video path does not exist on disk."""


class GpuLockTimeout(TimeoutError):
  """Timed out waiting to acquire the cross-process GPU file lock."""
