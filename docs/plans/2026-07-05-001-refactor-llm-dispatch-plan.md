---
title: "refactor: Extract LLM backend selection + dispatch truth-table tests"
type: refactor
status: active
date: 2026-07-05
trello: https://trello.com/c/wBqoCsDp
---

# refactor: Extract LLM backend selection + dispatch truth-table tests

## Summary

Extract a single `_select_backend()` decision function from the divergent 6-branch dispatch ladders in `LLM._generate_primary` / `LLM._agenerate_primary`, pin the full dispatch semantics with a truth-table test, and fix four bugs the ladder currently hides (silent tool-drop, lossy fallback constructors, deprecated asyncio busy-poll, missing async codex branch).

---

## Problem Frame

The real complexity of `merceka_core/llm.py` is not its well-tested pure helpers — it is dispatch *ordering*, which exists twice (sync at `_generate_primary`, async at `_agenerate_primary`), has already diverged (async has no codex branch), and is untested. Four confirmed bugs live only in that ordering.

---

## Requirements

- R1. One decision function determines the backend for a (provider flags, tools, allowed_tools, fallback) combination; both sync and async ladders consume it, so they cannot diverge again.
- R2. **B1 fixed:** Claude + Python-callable tools + no `allowed_tools` + no fallback must raise a clear `ValueError` instead of silently dropping the tools. (Current behavior is silently wrong — no caller can be depending on it correctly.)
- R3. **B3 fixed:** every fallback-LLM constructor preserves `add_dirs` and `allowed_tools`; the `stream_generate` one-chunk constructor additionally preserves `tools` and `max_tool_rounds`.
- R4. **B2 fixed:** `astream_generate` uses `asyncio.get_running_loop()` + an async-native handoff (no 20ms busy-poll on a sync `queue.Queue`).
- R5. **B4 fixed:** async ladder gains the missing codex branch (`asyncio.to_thread(self._codex_call)`); `gemini/` models calling plain `generate`/`agenerate`/`chat` raise a clear `ValueError` naming `generate_with_video`/grounding instead of falling through to Ollama with a confusing model-not-found.
- R6. A dispatch truth-table test pins every cell: prefix (claude//codex//gemini//openrouter-substring/local) × has-tools × has-allowed_tools × has-fallback, for sync and async. Includes the substring semantics of `use_openrouter` (`"openrouter" in name`, not prefix).
- R7. Public API unchanged: flag attributes (`use_claude`, `use_codex`, `use_gemini`, `use_openrouter`) survive as attributes; `tests/contracts/` (mindweaver, slab) pass unmodified.

---

## Scope Boundaries

- `chat()`, `_run_tool_loop`, `generate_with_resource` keep their internal mini-ladders for now — unifying them onto `_select_backend` belongs to the split card (ctcHvgCe). Exception: the `gemini/` guard (R5) applies to `chat()` too if trivial.
- No gemini *text-generation* implementation — out of scope; R5 makes the gap explicit instead of silent.
- Retry policy and the caught-exception tuple of the fallback cascade are untouched (D7/D10 in `docs/archive/evaluation_plan.md` era decisions; `VideoNotFoundError ⊂ FileNotFoundError` interplay is load-bearing for the cascade and pinned by slab contract tests).

---

## Context & Research

- Dispatch semantics, branch precedence, and both known fallthrough cells mapped by repo research (sync ladder `_generate_primary`; async twin `_agenerate_primary`; stream two-branch ladder; `chat()` mini-ladder).
- Flag semantics: `use_claude` = `startswith("claude/")`; `use_codex` = `startswith("codex/")`; `use_gemini` = `startswith("gemini/")`; `use_openrouter` = none-of-the-above AND `"openrouter" in model_name` (substring). Local Ollama is the fallthrough; `_verify()` runs only on the local path.
- Test conventions to follow: plain pytest + `monkeypatch`, 2-space indent; subprocess providers mocked via `monkeypatch.setattr("merceka_core.llm.subprocess.run", ...)`; OpenRouter via fake `urlopen`/`httpx.AsyncClient`; construct local-path LLMs offline via `patch.object(LLM, '_verify')`. Seed truth-table assertions exist in `tests/test_claude_provider.py::TestProviderDetection` and `tests/contracts/test_slab_surface.py`.
- Institutional: propagate errors, don't return neutral values (B1 fix rationale); asyncio house style is `asyncio.to_thread` / `run_in_executor` + `asyncio.wait_for`.

---

## Key Technical Decisions

- **`_select_backend()` returns a small enum-like decision** (e.g. string constants or `enum.Enum`: `CLAUDE`, `CLAUDE_FALLBACK_FOR_TOOLS`, `CODEX`, `TOOL_LOOP`, `OPENROUTER`, `LOCAL`, plus `GEMINI_UNSUPPORTED` → raise), computed from `(use_* flags, bool(_tool_schemas), bool(allowed_tools), bool(fallback))`. The ladders become thin `match`/dict dispatch over the decision. Rationale: a value-returning pure function is directly truth-table-testable without mocking any transport.
- **B1 raises `ValueError`** (not warn): silently-wrong edge, propagate-don't-swallow. Message names the three escape hatches (pass `allowed_tools`, set `fallback`, or drop `tools`).
- **B3 via one `_fallback_llm()` helper** constructing the fallback `LLM` with the full kwarg set — four call sites share the pattern today (extract-on-second-occurrence, we're at four).
- **B2 via `asyncio.Queue` + `loop.call_soon_threadsafe`** from the producer thread; `None` sentinel terminates. No busy-poll.
- **Gemini plain-generate raises** rather than routing to Ollama: explicit gap beats confusing downstream 404.

---

## Open Questions

### Resolved During Planning
- Warn vs raise for B1: raise (see decisions).
- Gemini fallthrough: raise with pointer to `generate_with_video`/grounding.

### Deferred to Implementation
- Whether `chat()`'s gemini guard is a one-liner or needs restructuring — implement if trivial, else note as follow-up on the split card.

---

## Implementation Units

- U1. **Characterization truth-table test (current behavior)**

**Goal:** Pin today's dispatch semantics for every cell that will NOT change, before touching the ladder.
**Requirements:** R6, R7
**Dependencies:** None
**Files:**
- Create: `tests/test_dispatch_truth_table.py`
**Approach:** Parametrized test over (model_name prefix × tools × allowed_tools × fallback) asserting which transport gets hit, using monkeypatched `subprocess.run`, `urlopen`, `httpx.AsyncClient`, and ollama client per existing conventions. Cells whose behavior changes in U2–U4 are written as the NEW expected behavior and marked `xfail(strict=True)` until the fix lands, then the marks drop.
**Execution note:** Test-first — this unit lands red-with-xfails before any production change.
**Test scenarios:**
- Happy path: `claude/x` no tools → `_claude_call`; `codex/x` sync → `_codex_call`; `openrouter/y` → OpenRouter POST; plain `z` → Ollama; tools + openrouter → tool loop.
- Edge: `"my-openrouter-proxy"` (substring, no prefix) → OpenRouter; flag attributes match slab contract pins.
- Edge (xfail→fix): `codex/x` **async** → `_codex_call` (today: Ollama); `gemini/x` `.generate()` → ValueError (today: Ollama).
- Error path (xfail→fix): claude + tools + no allowed_tools + no fallback → ValueError (today: silent drop).
**Verification:** Suite green on main with xfails; xfails flip to passes as U2–U4 land; zero xfails at card end.

- U2. **Extract `_select_backend()`; add async codex branch; gemini guard**

**Goal:** Single decision function consumed by both ladders (R1, R5).
**Requirements:** R1, R5, R7
**Dependencies:** U1
**Files:**
- Modify: `merceka_core/llm.py`
- Test: `tests/test_dispatch_truth_table.py`
**Approach:** Pure method `_select_backend()` → decision constant; `_generate_primary`/`_agenerate_primary` become mechanical maps from decision → call. Async map adds `CODEX → asyncio.to_thread(self._codex_call)`. `GEMINI_UNSUPPORTED` raises in both; apply same guard in `chat()` if a one-liner.
**Patterns to follow:** existing private-method naming; module constants style (`_RETRY_*`).
**Test scenarios:** covered by U1 table (xfail flips); plus direct unit tests of `_select_backend()` returning each decision for representative flag combos.
**Verification:** Both ladders contain no provider conditionals except the decision map; contract tests pass.

- U3. **Fallback constructor fidelity (B1 + B3)**

**Goal:** Raise on silent tool-drop; fallback LLMs inherit full config.
**Requirements:** R2, R3
**Dependencies:** U2
**Files:**
- Modify: `merceka_core/llm.py`
- Test: `tests/test_dispatch_truth_table.py`
**Approach:** `_fallback_llm()` helper builds `LLM(fallback_or_name, system_prompt, think, output_schema, tools, max_tool_rounds, add_dirs, allowed_tools)`; replace the four bespoke constructors (generate/agenerate outer catch + claude-tools branches) and the stream one-chunk constructor. B1: when decision is `CLAUDE` and `_tool_schemas` non-empty and no `allowed_tools` and no `fallback` → raise ValueError.
**Test scenarios:**
- Happy: claude+tools+fallback → fallback LLM receives add_dirs/allowed_tools/tools (assert via captured constructor kwargs).
- Error: claude+tools+no-fallback+no-allowed_tools → ValueError message names escape hatches.
- Edge: outer-catch fallback (OpenRouter 5xx → fallback) preserves add_dirs/allowed_tools; stream one-chunk path preserves tools/max_tool_rounds; stream self-wrap when `fallback is None` keeps current behavior (pinned).
**Verification:** All four B3 sites route through the helper (grep: one `LLM(` construction inside fallback paths).

- U4. **Modernize `astream_generate` (B2)**

**Goal:** Async-native streaming handoff.
**Requirements:** R4
**Dependencies:** U2
**Files:**
- Modify: `merceka_core/llm.py`
- Test: `tests/test_dispatch_truth_table.py` (or `tests/test_llm_stream.py` if cleaner)
**Approach:** `asyncio.get_running_loop()`; producer thread pushes chunks via `loop.call_soon_threadsafe(q.put_nowait, chunk)` into `asyncio.Queue`; `None` sentinel ends; consumer `await q.get()`. Exceptions from the thread are forwarded and re-raised.
**Test scenarios:**
- Happy: async iteration yields the same chunk sequence as sync `stream_generate` (mock claude stream).
- Error: producer exception surfaces to the async consumer, not swallowed.
- Edge: no `DeprecationWarning` under `-W error::DeprecationWarning`.
**Verification:** No `queue.Queue`/`sleep(0.02)` in the method; test proves chunk parity and exception propagation.

---

## System-Wide Impact

- **API surface parity:** flag attributes and all public method signatures unchanged; new ValueErrors only on cells that were silently broken (B1) or misrouted (gemini→Ollama).
- **Error propagation:** B1/R5 convert silent misbehavior into eager, message-rich errors — downstream consumers (mindweaver/slab/videototext/instascraper/quest) verified to not exercise those cells.
- **Unchanged invariants:** fallback catch-tuple, retry-before-fallback layering, `VideoNotFoundError ⊂ FileNotFoundError` cascade interplay, `stream_generate` self-wrap behavior.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| A downstream caller relies on gemini→Ollama fallthrough or silent tool-drop | Grepped all six consumer repos: no `gemini/` plain-generate or claude+tools-no-fallback call sites; ValueErrors are eager and descriptive |
| Truth-table mocks drift from real transports | Mocks follow the exact conventions of existing green tests; integration tests still cover real transports |
| Async queue refactor changes chunk timing | Parity test asserts identical chunk sequence; timing is not part of the contract |

---

## Sources & References

- Trello: https://trello.com/c/wBqoCsDp
- Related code: `merceka_core/llm.py` (`_generate_primary`, `_agenerate_primary`, `stream_generate`, `astream_generate`, `chat`)
- Contract pins: `tests/contracts/test_slab_surface.py`, `tests/contracts/test_mindweaver_surface.py`
- Seed detection tests: `tests/test_claude_provider.py::TestProviderDetection`
