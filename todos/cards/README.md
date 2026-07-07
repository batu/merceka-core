# merceka_core improvement cards — load procedure

These are the survivors of a Fable pass (2026-07-06). They are **on file only** — no
Trello board is claimed and no cards are posted. A later Opus/Codex *implementation*
session claims a board and loads them. Fabled ≠ implemented.

## What's here
`NN-<slug>.md`, NN = recommended execution order.

- **01–08 are health briefs** (fix/robustness/simplify/refactor): self-contained,
  direct-to-work, with `file:line` evidence, a decided approach, a scope fence,
  acceptance criteria, and an exact verification command carrying today's stamped
  baseline. Work them top-to-bottom.
- **09–11 are blue-sky idea cards**: lighter; each says "brainstorm/plan before
  implementing; re-verify repo facts at pickup." Do not treat them as ready-to-code.

## Shared-library gate (READ THIS)
merceka_core is a **foundation library** imported by other repos — the mindweaver
backend imports it **at startup**, and slab/videototext depend on its public surface.
`tests/contracts/` (17 tests as of 2026-07-06) pins the consumer-facing shapes.
Every card that touches public API (`__init__.py` exports, `LLM` construction/signatures,
agent providers, `gpu_lock`, the exception hierarchy, or the install/dependency surface)
carries a **"consumers still import clean"** gate:

```
uv run pytest tests/contracts/ -q   # must stay green (baseline: 17 passed)
uv run python -c "import merceka_core; from merceka_core import LLM, Agent, gpu_lock, ClaudeCodeAgentProvider, CodexAgentProvider, VideoUploadError; print('OK')"
```
Treat a red contract test as blocking. Any public-API *break* (not just addition) is
high-risk: land it only with the gate green and note it in the handoff.

## Baselines (stamped 2026-07-06, main @ 1e859de)
- Full suite: `uv run pytest tests/ -q` → **346 passed, 1 skipped, 6 deselected** (~4.5s)
- Contract gate: `uv run pytest tests/contracts/ -q` → **17 passed**
- Re-stamp both at pickup; line numbers and counts drift.

## Dependencies between cards
- **06 Depends_on 03** — both edit `agents/{claude_code,codex,pi}.py`. Land 03 (the
  correctness fix: relocate the stdin write) first, then 06 (dedup) on top.

## Load procedure (implementation session)
1. Claim a scratch board: `twf board claim`, commit the claim. Trello creds live in
   `utolye/.env` — export them into **every** twf invocation (worker worktrees resolve
   the wrong `.env` otherwise).
2. POST each `NN-*.md` to Trello `/1/cards`: first `# ` heading → card `name`, the rest
   of the file → `desc` (verbatim). Post in file order so `Depends_on` slugs resolve to
   real shortids; wire 06's `Depends_on: 03-...` to card 03's shortid.
3. Health cards (01–08) → **Todo**. Idea cards (09–11) → **Ideas** (or the bottom of
   Todo).
4. Then run the normal twf pipeline. No PRs — the conductor merges; out-of-fence needs
   go in the handoff SURPRISES section.

## Rejected (do not re-propose without new evidence)
See `docs/ideation/2026-07-06-fable-pass.md` for the full rejection table. Headline
kills: golden-prompt regression harness and public-API snapshot/deprecation tooling
(enterprise QA/semver machinery at N=1 maintainer), response caching (speculative),
model-alias registry (unconfirmed demand), Gemini-upload SIGKILL recovery (TTL
self-heals), and `VideoNotFoundError`-inherits-`FileNotFoundError` (that's
**intentional and contract-tested**, not a bug).
