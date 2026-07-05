---
title: "refactor: split llm.py along its natural seams"
type: refactor
status: active
date: 2026-07-05
trello: https://trello.com/c/ctcHvgCe
---

# refactor: split llm.py along its natural seams

## Summary

Extract three cohesive modules from the 1433-line `llm.py` — message/schema builders, retry policy, and the Gemini surface — leaving `llm.py` as the `LLM` class + its dispatch. Every existing import keeps working via re-exports.

## Requirements

- R1. `messages.py`: `create_message`, `create_message_with_resource`, `create_ollama_vision_message`, `tool_from_callable`, `_python_type_to_json`, `_parse_param_docs`, `OutputSchema`, `_schema_name`, `_openrouter_response_format`.
- R2. `retry.py`: `_RETRY_*` constants, `_retry_delay`, `_retry_after_seconds`.
- R3. `llm_gemini.py`: `_gemini_client`, `_gemini_poll_until_active`, `_build_video_config`, `_gemini_video_call`, `_extract_grounding`, `_generate_with_search_grounding_sync` and the rest of the Gemini block (llm.py:1180-end).
- R4. `llm.py` re-exports every moved public name (and the private names tests import) so `from merceka_core.llm import X` keeps working for all current X; `__all__` unchanged. Contract tests and all existing tests pass unmodified.
- R5. No logic changes — move-only, verified by AST-level equivalence of moved functions.

## Scope Boundaries
- No unification of `chat()`/tool-loop mini-ladders (noted as future work in dispatch plan).
- No renames of moved symbols.
- OpenRouter transport and tool loop stay in `llm.py` (they are entangled with `LLM` state; extracting them is not a clean seam today).

## Implementation Units
- U1. `messages.py` + re-exports; suite green.
- U2. `retry.py` + re-exports; suite green (test_retry_policy imports from llm — keeps working).
- U3. `llm_gemini.py` + re-exports; suite green (slab contract pins `generate_with_search_grounding` from llm).
- U4. Verification: `wc -l` target ≤ ~900 for llm.py; grep proves no moved-symbol definitions remain in llm.py; full suite + ruff.

## Verification
- Full suite green with zero edits to existing tests; ruff clean; import smoke of all downstream-pinned names.
