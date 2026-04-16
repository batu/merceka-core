# merceka_core

Core utilities for merceka projects.

## Installation

```bash
pip install merceka_core
```

## Usage

```python
from merceka_core.llm import LLM

# Local Ollama model
llm = LLM("gemma3:27b")
response = llm.generate("Hello, how are you?")

# Cloud model via OpenRouter
llm = LLM("openrouter/google/gemini-2.5-flash-lite-preview-09-2025")
response = llm.generate("Hello, how are you?")

# OpenRouter vision with Claude Sonnet
llm = LLM("openrouter/anthropic/claude-sonnet-4-5")
description = llm.generate_with_resource("What's in this image?", "frame.png")

# Gemini long-context video (needs GOOGLE_API_KEY)
# `gemini-flash-latest` is the recommended default for video: it aliases
# the newest full-fat Flash (currently gemini-3-flash-preview, Dec 2025)
# and auto-upgrades as newer Flash models ship. Use `gemini-pro-latest`
# when you need Pro-tier reasoning on a curated clip.
llm = LLM("gemini/gemini-flash-latest")
summary = llm.generate_with_video(
    "Summarize the mechanic shown in this gameplay footage.",
    "playthrough.mp4",
    timeout_s=300,  # upload + poll-until-ACTIVE budget
)
```

### Cross-process GPU serialization

Multiple processes (slab vision triage, mindweaver enrichment, ad-hoc
CLI runs) share one GPU. Wrap GPU work in `gpu_lock()` — a file lock at
`~/.local/state/utolye/gpu.lock`. The kernel releases the fd on
process death, so there is no stale-lock cleanup.

```python
from merceka_core import gpu_lock

async def transcribe(audio_path):
    async with gpu_lock(timeout=600):
        return await whisperx.transcribe(audio_path)
```

### Exception hierarchy

`merceka_core.errors` exposes four classes so downstream consumers can
distinguish retryable from terminal failures without importing SDK
types:

- `VideoUploadError` — codec/size/quota rejection; terminal.
- `VideoBackendError` — transient 5xx during inference; retryable.
- `VideoNotFoundError(FileNotFoundError)` — path missing.
- `GpuLockTimeout(TimeoutError)` — `gpu_lock` timeout.

## Development

This project uses [nbdev](https://nbdev.fast.ai/) for literate programming.

```bash
# Install dependencies
uv sync

# Start Jupyter
uv run jupyter lab

# Export notebooks to Python modules
uv run nbdev_export
```

