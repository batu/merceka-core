# merceka_core

Core utilities for merceka projects.

## Installation

```bash
pip install merceka-core
```

## Usage

```python
from merceka_core import LLM

# Local Ollama model
llm = LLM("gemma3:27b")
response = llm.generate("Hello, how are you?")

# Cloud model via OpenRouter
llm = LLM("openrouter/google/gemini-2.5-flash-lite-preview-09-2025")
response = llm.generate("Hello, how are you?")

# OpenRouter vision with Claude Sonnet
llm = LLM("openrouter/anthropic/claude-sonnet-4-5")
description = llm.generate_with_resource("What's in this image?", "frame.png")

# Gemini Flash image understanding (needs GOOGLE_API_KEY) — cheap bulk
# vision analysis; same generate_with_resource API as OpenRouter/Ollama.
llm = LLM("gemini/gemini-flash-latest")
label = llm.generate_with_resource("What's in this image?", "frame.png")

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

### Credentials

Provider keys come from the environment. Importing `merceka_core.llm` (or
`llm_gemini`) also fills the library's own keys (`OPENROUTER_API_KEY`,
`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GEMINI_API_KEY`, `FAL_KEY`, and the
OpenRouter header settings) from the nearest `.env` above the package. It fills
only keys that are absent, and never any other entry in that file.

- To disable a provider, set its key to an empty string (`OPENROUTER_API_KEY=`).
  `env -u` does not work, because an absent key is filled from `.env`.
- `PYTHON_DOTENV_DISABLED=1` skips `.env` entirely.
- CLI subprocesses (Claude Code, Codex, pi, grok) never inherit credentials. They
  authenticate with their own logins.

### Images

```python
from merceka_core.image import generate_image, edit_image, inpaint, upscale_image

img = generate_image("a red kite on a white background", model="openai/gpt-image-2",
                     transparent=True)
edited = edit_image(img, "make the kite blue", model="google/gemini-3.1-flash-image-preview")
filled = inpaint(img, mask, "clear sky")          # default fal-ai/flux-pro/v1/fill
big = upscale_image(img, scale=2.0)               # default fal-ai/esrgan
```

Model ids pick the provider: `openai/...` and `google/...` go direct when their
key is set, otherwise through OpenRouter; `fal-ai/...` goes to fal. Edits that
cannot keep the input's aspect ratio raise an error instead of stretching the
image.

`inpaint(..., reference_images=[sheet])` sends extra images after the edited
one (images 2, 3, ...), for example a character reference sheet. The mask
applies to the first image only. Only `openai/` models accept references; other
providers raise before the call.

### Cost ledger

Every metered provider call appends one JSONL row to `~/.merceka/costs.jsonl`
(override with `MERCEKA_COST_LEDGER`). `usd` is the provider's own figure
(`usd_source: "provider"`) or rate-table arithmetic from `rates.json`
(`usd_source: "rates"`, an estimate). It is `null` when neither exists.

```python
from merceka_core import costs

with costs.attribution({"sessionId": "level_42", "operation": "extract"}):
    ...  # every row recorded in here carries this meta
```

```bash
uv run python -m merceka_core.costs --since 2026-09-30T00:00:00+03:00
```

The summary reports `usd_metered` and `usd_rates` separately; `usd_known` is their sum.

### Vision critique

```python
from merceka_core.vision import critique

result = critique(["ours.png"], reference="target.png")
result["verdict"], result["score"], result["defects"], result["recurring_checks"]
```

A panel of judges (OpenRouter models, the local `claude` and `codex` CLIs, and an
Anthropic zoom judge) scores the images. The verdict aggregates the judges that
answered; `skipped` lists the ones that didn't.

### Agents

```python
from merceka_core import Agent, AgentRequest, ClaudeCodeAgentProvider

agent = Agent(ClaudeCodeAgentProvider(model="sonnet"))
result = await agent.run(AgentRequest(message="Summarise chapter 3", system_prompt="",
                                      roots=(book_dir,)))
```

`AgentProfile.READ_ONLY` (the default) restricts the agent to read and search
tools inside `roots`. `AgentProfile.WRITE` allows edits and shell commands. The
Claude, Codex and pi providers enforce this with each CLI's own tool and sandbox
flags.

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

Plain Python package — edit the modules under `merceka_core/` directly.
(The repo previously used nbdev; the notebook layer was removed 2026-07-05
after it drifted from the hand-edited `.py` files.)

```bash
# Install dependencies
uv sync

# Run the default (non-integration) test suite
uv run pytest tests/
```
