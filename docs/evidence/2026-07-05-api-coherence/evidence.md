---
status: passed
subject: coherent top-level API with lazy exports (card vaYo0Pdk, PR #9)
created: 2026-07-05
mode: pipeline
---

# Evidence: top-level API coherence

## Verdict
All downstream-critical names now import from `merceka_core` while `import merceka_core` stays light; reviewer verified zero consumer breakage across all six sibling repos and PEP 562 correctness (identity, caching, pickling).

## Evidence Captured
| Type | Artifact / Command | Result |
|------|--------------------|--------|
| test | `uv run pytest tests/` | 335 passed (7 new) |
| lightness | subprocess: `import merceka_core` and touching `PiAgentProvider` | litellm/ollama/google-genai absent both times |
| consumer sweep | reviewer grep of 6 sibling repos | purely additive, no reflection patterns, contract tests green |

## Reviewer Assessments
| Reviewer | Status | Result |
|----------|--------|--------|
| ce-api-contract-reviewer | passed | 1 informational P3 (star-import now resolves the heavy stack — inherent, no star-import consumers exist); provider-lightness gap closed with a test |

## Gaps
- None blocking.

## Next Action
None
