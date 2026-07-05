---
title: "test: cover the untested giants — evaluation.py, Gemini surface, image.py dispatch"
type: test
status: active
date: 2026-07-05
trello: https://trello.com/c/8pmzZzqb
---

# test: cover the untested giants

## Summary
Add default-suite (non-integration) unit tests for the three largest untested bodies of code: `evaluation.py` (~800 lines, zero tests), the Gemini surface in `llm_gemini.py` (integration-marker-only), and `image.py`'s HTTP dispatch.

## Requirements
- R1. `tests/test_evaluation.py`: `_detect_calling_convention` (all 4 conventions + error cases), `config_name`, `Evaluation`/`TaskResult`/`ExperimentResults` round-trip (to_dict/from_dict/save/load), slicing (`by_task`/`by_config`/`failures`/`successes`), `success_rate` (bool-only counting), `run_experiment` happy path with stub Task/Evaluator (no LLM).
- R2. `tests/test_llm_gemini.py`: `_gemini_poll_until_active` (ACTIVE, FAILED→VideoUploadError, timeout→VideoUploadError, poll loop calls files.get), `_build_video_config` kwarg translation, `_gemini_video_call` happy path + missing file→VideoNotFoundError + retry-on-5xx + file cleanup via fake client, `_extract_grounding` shapes (full metadata, missing metadata → empty lists).
- R3. `tests/test_image_dispatch.py`: `generate_image`/`edit_image` OpenAI-vs-OpenRouter branch selection with transports mocked (which endpoint gets hit per model string), error propagation on non-200.
- R4. All tests in the default suite (no `integration`/`gpu` markers), fully offline, following repo conventions: plain pytest + monkeypatch, 2-space indent, fake clients over unittest.mock where the neighboring tests do that.

## Scope Boundaries
- No production-code changes except bugs found while writing tests (report, fix minimally, note in PR).
- gpu.py cross-process test stays `@gpu` (needs real fcntl across processes); not in scope.

## Verification
- Full suite green; new files run offline (`uv run pytest tests/test_evaluation.py tests/test_llm_gemini.py tests/test_image_dispatch.py`); ruff clean.
