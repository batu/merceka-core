---
status: passed
subject: revive silent-dead git_hash + evaluation extra (card z6T92iUS, PR #5)
created: 2026-07-05
mode: pipeline
---

# Evidence: eval-deps fix

## Verdict
The reproducibility field works for the first time: a live `ExperimentResults` now records the actual HEAD SHA, and the pandas dependency is declared and error-guarded.

## Evidence Captured
| Type | Artifact / Command | Result |
|------|--------------------|--------|
| live smoke | `ExperimentResults(...).git_hash` | real HEAD SHA `64db1049fbb9…` (was permanently `"unknown"`) |
| test | `uv run pytest tests/` | 251 passed, 1 skipped |
| test | new `tests/test_evaluation_deps.py` | 7 cases: in-repo SHA, outside-repo, unborn HEAD, no git binary, extra metadata, ImportError message, works-without-GitPython |
| lint | `uv run ruff check .` | clean |

## Reviewer Assessments
| Reviewer | Status | Result |
|----------|--------|--------|
| ce-correctness-reviewer | passed after fixes | 0 production findings; 3 P3 test-robustness nits, all fixed |

## Gaps
- None.

## Next Action
None
