---
status: partial
subject: merceka_core.vision critique
created: 2026-07-07
mode: pipeline
---

# Evidence: merceka_core.vision critique

## Verdict
The new vision critique behavior is covered by focused mocked tests and broad regression tests, but the exact full-suite command is blocked by a pre-existing GPU lock timeout test hang.

## What Changed
- Added `merceka_core.vision.critique` with OpenRouter multimodal judge calls, structured JSON request controls, tolerant parsing, skip accounting, median scoring, consensus keys, and budget checks.
- Added `openrouter_budget_floor()` using OpenRouter credits balance.
- Added mocked tests for parsing, clamping, aggregation, skip classes, budget behavior, zero participants, malformed envelopes, and multi-image repeated-key defects.

## Evidence Captured
| Type | Artifact / Command | Result |
|------|--------------------|--------|
| test | `uv run pytest -q tests/test_vision_critique.py` | 31 passed |
| test | `uv run pytest -q -k 'not test_gpu_lock_timeout_raises'` | 376 passed, 1 skipped, 7 deselected |
| lint | `uv run ruff check .` | passed |
| format | `uv run ruff format --check merceka_core/vision tests/test_vision_critique.py` | passed |
| whitespace | `git diff --check` | passed |

## Reviewer Assessments
| Reviewer | Status | Result |
|----------|--------|--------|
| ce-code-review | partial | Safe parser, request-shape, aggregation, and test-coverage fixes were applied. Remaining notes conflict with explicit contract choices or are broader policy changes. |

## Analysis
`uv run pytest -q` was attempted and then rerun with `-vv -x`; it blocks in `tests/contracts/test_mindweaver_surface.py::test_gpu_lock_timeout_raises`, before the new vision tests run. Running that test alone also hangs. Excluding only that test allows the rest of the default-selected suite to pass.

## Gaps
- Exact requested `uv run pytest -q` is not green because of the pre-existing GPU lock timeout hang.
- Full-repo `ruff format --check .` reports many pre-existing files would be reformatted; changed files pass format check.

## Next Action
Fix or quarantine the existing GPU lock timeout test so the repository-wide `uv run pytest -q` command can complete without exclusions.
