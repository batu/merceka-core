---
title: "feat: Gemini Flash image understanding via generate_with_resource"
type: feat
status: active
date: 2026-07-05
trello: https://trello.com/c/OJ3yIghn
---

# feat: Gemini Flash image understanding

## Summary
`LLM("gemini/gemini-flash-latest").generate_with_resource("what's in this?", "img.png")` works: a gemini branch in the existing vision entry point routes to a new `_gemini_image_call` in `llm_gemini.py` using inline image bytes (no upload/poll ceremony — images fit inline, unlike video).

## Problem Frame
Consumers (instascraper, slab, mindweaver) want cheap bulk image analysis on Flash. Today `generate_with_resource` on a `gemini/` model silently falls through to the *Ollama* path — the same misroute class the dispatch card fixed for plain generate. Extending the real entry point (not adding a parallel `understand_image()` API) keeps one vision surface.

## Key Technical Decisions
- **Extend `generate_with_resource`/`agenerate_with_resource`** with a `use_gemini` branch — consumers keep one vision API across providers; no new public function.
- **Inline bytes** (`types.Part.from_bytes`) not the Files API — images are small; skips upload/poll/delete entirely.
- **Mirror the video path's structure**: module function in `llm_gemini.py` taking `llm`, parsing via `llm._parse_response`, retrying via shared `retry.py` helpers, raising `VideoBackendError` on transport failure (the shared gemini-transport error despite its video-era name) and plain `FileNotFoundError` for missing files (keeps the fallback cascade semantics).
- Model comes from `self.model_name` (`gemini/` prefix stripped) — README's `gemini-flash-latest` alias is the recommended default; no hardcoded model.

## Implementation Units
- U1. `llm_gemini.py::_gemini_image_call(llm, message, resource_path, **kwargs)` — read bytes, mime via `mimetypes` + image fallback map, `generate_content` with retry, parse via `llm._parse_response`. Test: fake client — happy path, output_schema parse, retry-on-5xx, non-retryable raise, missing file, mime detection.
- U2. Gemini branches in `generate_with_resource` (sync) and `agenerate_with_resource` (async, `asyncio.to_thread`) + docstring updates. Test: dispatch pins — `gemini/` model hits the gemini client, not Ollama; async mirrors sync; contract signatures unchanged (mindweaver pins agenerate_with_resource params).
- U3. README usage snippet.

## Scope Boundaries
- No PDF/video handling changes; no streaming; no new top-level API.
- Multi-image lists deferred (single `resource_path`, matching the existing signature).

## Verification
- Full suite green; new tests offline via fake client; contract tests unmodified.
