# idea: consolidate scattered model-string prefix parsing into one parse_model_ref

> Blue-sky card — brainstorm/plan before implementing; re-verify repo facts at pickup
> (baselines and line numbers drift). Confidence ~65%, complexity S.

## Summary
`LLM.__init__` derives the `use_claude`/`use_codex`/`use_gemini`/`use_openrouter`
boolean quartet from `model_name.startswith(...)` (llm.py ~146-150), and the same
`"provider/model"` grammar is then re-parsed as ad hoc `removeprefix("openrouter/")`,
`removeprefix("claude/")`, `removeprefix("codex/")` calls scattered across the file
(around lines 549/724/757/787). Consolidate into one small frozen dataclass
`parse_model_ref(model_name) -> ModelRef(provider, model)` computed once in `__init__`,
replacing the scattered `removeprefix` calls and giving consumers a documented,
testable grammar for the prefix convention instead of "read the source to learn the
prefixes."

Note: the sync/async *dispatch* is already well-factored via `_select_backend()`
(2026-07-05, PR #4) — do NOT touch that. This card is only about the string-parsing
that feeds it, which is still duplicated.

## Why it matters (this scale)
Pure de-duplication with a consumer-facing bonus: a single parsed value replaces 4+
scattered string ops and the boolean quartet, and documents the `"provider/model"`
convention the ~4 consumer repos rely on. Low cost, behavior-preserving.

## Watch-outs
- Behavior must stay identical — this is a refactor, not a semantic change. The
  contract tests (which assert `use_openrouter`/`use_claude`/`use_gemini` flags on
  constructed `LLM`s) must stay green; keep those attributes (or derive them from the
  parsed ref without changing their truth values).
- Public-API-adjacent: `LLM(...)` construction is consumer surface — run the
  `tests/contracts/` + import gate. Don't over-build a provider registry; a small
  parse function + dataclass is the whole scope.

## Complexity / confidence
S · ~65%. Adjacent to (but narrower than) a model-alias registry, which was rejected as
unconfirmed-demand at this scale.
