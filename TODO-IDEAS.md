# TODO-IDEAS — merceka_core (Fable pass 2026-07-06)

Ranked improvement cards from a Fable audit/ideation pass. **These live as files; they
are not on a board.** Do not implement straight from this list — load them onto a twf
board first per `todos/cards/README.md` (claim board, POST each `todos/cards/NN-*.md`,
health → Todo, ideas → Ideas), then run the normal pipeline.

**Shared-library note:** merceka_core is imported by other repos (mindweaver backend
imports it *at startup*; slab/videototext depend on its public surface). Every card
touching public API carries a "consumers still import clean" gate — keep
`tests/contracts/` green (baseline: 17 passed) and run the import smoke.

Baselines stamped 2026-07-06 (main @ 1e859de): `uv run pytest tests/ -q` →
**346 passed, 1 skipped, 6 deselected**. Re-verify at pickup.

## Health (direct-to-work briefs)
1. **01-fix-chat-bypasses-select-backend** — `chat()` has no `use_codex` branch; codex/
   and claude+tools chats misroute to Ollama. **P1.**
2. **02-robustness-gpu-lock-executor-thread-leak** — timed-out `gpu_lock` leaks a
   shared-executor thread; enough timeouts hang the whole process. Fix: `LOCK_NB` poll.
3. **03-robustness-subprocess-stdin-cleanup-and-stream-timeout** — stdin written before
   the cleanup try in 3 providers (leaks child on `BrokenPipeError`); `_claude_stream`
   has no timeout/terminate.
4. **04-robustness-webhook-signature-and-body-cap** — wa_bot webhook POST has no
   `X-Hub-Signature-256` and no body-size cap → unauthenticated LLM-spend + DoS.
5. **05-simplify-remove-dead-litellm-and-chat-one** — drop the heavy litellm dep
   (imported only to silence its own noise) and the unused `_chat_one`.
6. **06-simplify-extract-shared-agent-subprocess-helpers** — de-dup ~250 identical
   lines across the 3 CLI providers into a flat helper module. *Depends_on 03.*
7. **07-robustness-wa-bot-client-transient-vs-terminal** — client collapses 429/5xx and
   400 both to `None` → silent reply loss; distinguish transient/terminal.
8. **08-refactor-document-image-and-evaluation-modules** — README omits `image.py` and
   `evaluation.py`; add `wa_bot`-style sections. Doc-only.

## Blue-sky ideas (brainstorm/plan first; re-verify repo facts at pickup)
9. **09-idea-usage-cost-accounting** (~85%) — OpenRouter `usage` is returned and
   discarded; attach per-call tokens/cost via a side-channel.
10. **10-idea-merceka-doctor-cli** (~85%) — one flat command checking env keys +
    provider/CLI reachability + gpu_lock writability.
11. **11-idea-parse-model-ref-consolidation** (~65%) — fold scattered `removeprefix`
    model-string parsing into one `parse_model_ref` dataclass (leave `_select_backend()`).

Full findings, survivors, and the rejection table:
`docs/ideation/2026-07-06-fable-pass.md`. Load procedure: `todos/cards/README.md`.
