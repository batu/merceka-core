---
status: passed
subject: LLM dispatch extraction + truth-table tests (card wBqoCsDp, PR #4)
created: 2026-07-05
mode: pipeline
---

# Evidence: LLM dispatch extraction

## Verdict
Fresh test-suite, live-behavior smoke, and lint runs confirm the dispatch refactor works and the four bug fixes behave as specified.

## What Changed
- `_select_backend()` extracted; sync/async ladders map over it (cannot diverge again)
- B1: CLI provider + Python tools + no escape → eager `ValueError` (was: silent tool drop)
- B3: all five fallback constructors preserve full config via `_fallback_llm()`
- B2: `astream_generate` async-native (no busy-poll, no deprecated `get_event_loop`), early-break bounded by stop event
- B4: async codex branch added; `gemini/` plain generate/chat raises with guidance
- Post-review: codex keeps its native `allowed_tools` escape (regression caught by ce-correctness)

## Evidence Captured
| Type | Artifact / Command | Result |
|------|--------------------|--------|
| test | `uv run pytest tests/` | 244 passed (45 truth-table), 1 skipped |
| smoke | live `LLM(...)` calls: gemini plain-generate, claude+tools no-escape, `_fallback_llm` fidelity | all three behave per spec |
| lint | `uv run ruff check .` | clean |
| contract | `tests/contracts/` (mindweaver, slab) | pass unmodified |

## Reviewer Assessments
| Reviewer | Status | Result |
|----------|--------|--------|
| ce-correctness-reviewer | passed after fixes | 2 P2 + 1 P3 found, all fixed and pinned with tests |
| ce-reliability-reviewer | not run | killed by session rate-limit; async-lifecycle territory covered by correctness pass |
| ce-kieran-python-reviewer | not run | killed by session rate-limit |

## Gaps
- Reliability/Python-style lenses did not complete (rate-limit). Correctness reviewer's pass covered the async lifecycle and dispatch-precedence risk areas; downstream call-site impact was verified during planning research (no consumer exercises the newly-raising cells).

## Next Action
None
