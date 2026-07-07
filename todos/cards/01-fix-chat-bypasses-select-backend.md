# fix: LLM.chat() bypasses _select_backend() and misroutes codex/ + CLI-tools calls to Ollama

`LLM.chat()` (`merceka_core/llm.py:245-269`) still carries its own hand-rolled
dispatch ladder that predates the 2026-07-05 `_select_backend()` truth-table
refactor (PR #4, `docs/solutions/2026-07-05-duplicated-dispatch-ladders-diverge-silently.md`).
Only `_generate_primary`/`_agenerate_primary` were migrated; `chat()` was missed.

Two confirmed misroutes (both reproduced by the audit, neither covered by a test):

1. **`codex/` models via `chat()`** — the ladder is `if self._tool_schemas: … elif
   self.use_claude … elif self.use_openrouter … else: self._local_call(...)`. There
   is **no `use_codex` branch**, so `LLM("codex/gpt-5.5").chat("hi")` (no tools)
   falls into the bare `else` and calls `ollama_chat(model="codex/gpt-5.5", ...)`
   against the local Ollama daemon — a nonsense model name and a confusing
   "model not found" far from the cause.
2. **claude/codex + native CLI tools via `chat()`** — when `_tool_schemas` is set,
   `chat()` unconditionally enters `_run_tool_loop` (`llm.py:445-466`), which only
   knows `use_openrouter` vs. `_local_call_raw`. A model constructed with
   `tools=`/`allowed_tools=` that `_select_backend()` correctly routes to
   `_BACKEND_CLAUDE`/`_BACKEND_CODEX` (llm.py:192-202) is instead sent to Ollama.

The individual transports are fine; only the *selection* inside `chat()` is wrong —
exactly the class of divergence the `_select_backend()` extraction was meant to make
structurally impossible. `generate()` is correct; `chat()` is the straggler.

## Decided approach
Route `chat()`'s backend choice through the existing `_select_backend()` decision
function (it already returns the right constant for prefix × tools × allowed_tools ×
fallback), then map the decision → transport the same mechanical way `_generate_primary`
does, preserving `chat()`'s history-append semantics (append user message before,
append assistant/`_response_to_history_content` after). Do **not** invent a parallel
ladder or duplicate the map — reuse the one that exists. The Gemini early-raise
(`llm.py:247-248`) already delegates to `_select_backend()`; extend that pattern to
the whole method.

Rejected: patching in a `use_codex` `elif` by hand — that just recreates the
divergence for the next provider. The whole point is one decision function.

## Scope fence
- `merceka_core/llm.py` — `chat()` only (and, if `_run_tool_loop` needs a CLI-tools
  path to satisfy the decision, the minimal branch there).
- `tests/test_dispatch_truth_table.py` — add `chat`-mode rows so codex/ and
  claude+allowed_tools selection is pinned for `chat()` the way it is for
  generate/agenerate.
- `tests/test_tool_calling.py` — extend beyond the current openrouter-only
  `chat()`+tools coverage.
Do not touch `generate()`/`agenerate()` (already correct) or other files.

## Acceptance criteria
- `LLM("codex/...").chat("x")` routes to the Codex CLI, not Ollama.
- A truth-table test asserts `chat()`'s selection matches `_select_backend()` for
  the codex and claude+allowed_tools cells; it fails on the current code.
- No behavior change for openrouter/claude/ollama `chat()` paths already covered.

## Verify (baseline stamped 2026-07-06, main @ 1e859de)
```
cd /home/batu/Desktop/utolye/merceka_core
uv run pytest tests/ -q            # baseline: 346 passed, 1 skipped, 6 deselected (~4.5s)
```
Consumer-import gate (shared lib — MUST stay green):
```
uv run pytest tests/contracts/ -q  # baseline: 17 passed
uv run python -c "import merceka_core; from merceka_core import LLM, Agent, gpu_lock, ClaudeCodeAgentProvider, CodexAgentProvider, VideoUploadError; print('OK')"
```

## Constraints
No PRs; the conductor merges. Out-of-fence needs → handoff SURPRISES. `chat()` is
public API used by consumers — the contract gate above is mandatory before landing.
