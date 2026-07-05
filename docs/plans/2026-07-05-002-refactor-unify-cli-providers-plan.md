---
title: "refactor: unify Claude/Codex CLI knowledge into one shared module"
type: refactor
status: active
date: 2026-07-05
trello: https://trello.com/c/0GzBmSu7
---

# refactor: unify Claude/Codex CLI knowledge into one shared module

## Summary

Extract the duplicated CLI knowledge (command building, env scrubbing, stream-json parsing) from `llm.py` and `agents/{claude_code,codex}.py` into one private module `merceka_core/_cli.py`, consumed by both layers.

## Problem Frame

Claude/Codex subprocess logic exists twice and has diverged. **Full delegation (LLM → agents/ providers) was investigated and rejected**: `AgentRequest` requires ≥1 existing root dir, always grants tool access via fixed profiles, and raises `ProviderFailure` (not in the LLM fallback cascade's catch tuple). Plain `LLM("claude/x").generate()` is a no-tools text call; delegating would silently grant Read/Grep/Glob on every call and break the cascade. The actual drift risk is the *CLI-flag and parsing knowledge*, so that is what gets one home.

## Requirements

- R1. One claude command builder, one codex exec command builder, one env scrub, one stream-json text-delta parser — in `merceka_core/_cli.py`.
- R2. `llm.py`'s `_claude_call`/`_claude_stream`/`_codex_call` and `agents/claude_code.py`/`agents/codex.py` consume them; behavior of each layer unchanged (byte-identical commands per layer).
- R3. Divergences become explicit builder parameters (ephemeral, images, cd, permission_mode, json_output), not parallel copies.
- R4. All existing tests pass unmodified (they pin each layer's exact commands).

## Implementation Units

- U1. **Create `merceka_core/_cli.py`** — `claude_command(...)`, `codex_exec_command(...)`, `claude_env()`, `claude_stream_text_delta(payload) -> str | None`, `is_claude_result_event(payload) -> bool`. Unit tests in `tests/test_cli_builders.py` pin each layer's exact command shape through the builders.
- U2. **Port `llm.py`** — `_claude_call`, `_claude_stream`, `_codex_call` build commands/env/parse via `_cli`. Existing tests (test_claude_provider, truth table) must pass unmodified.
- U3. **Port `agents/`** — `_command`/`_env`/`_text_delta_from_payload` in both providers via `_cli`. Existing provider tests pass unmodified.

## Scope Boundaries

- No behavior change in either layer; no new capabilities.
- The `codex --ephemeral` vs `--sandbox/--cd` split stays — they are different products (one-shot text vs rooted agent), now visible as parameters of one builder.

## Verification

- Full suite green with zero edits to existing test files.
- `grep -c '"claude", "-p"' merceka_core/` → only `_cli.py`; same for `"codex", "exec"` (test: no stray literal command lists outside `_cli.py`).
