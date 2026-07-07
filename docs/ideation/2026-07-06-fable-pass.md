# merceka_core — Fable pass, 2026-07-06

Conductor: Fable (Opus). Pass type: audit → filter → cards. **Zero product code edited**
— output is card files, this doc, and a root reminder. Fabled ≠ implemented.

- Repo: `/home/batu/Desktop/utolye/merceka_core` (remote `batu/merceka-core`)
- Start: clean tree on `main` @ `1e859de`, up to date with origin.
- Baselines stamped this pass: full suite `346 passed, 1 skipped, 6 deselected`
  (`uv run pytest tests/ -q`, ~4.5s); consumer-contract gate `17 passed`
  (`uv run pytest tests/contracts/ -q`).

## What this repo is
A **shared foundation library** for LLM calls used across a solo dev's constellation
(mindweaver backend — imports it *at startup* — plus slab, videototext, wa_bot). Wraps
OpenRouter (raw httpx), Ollama (local GPU), Gemini image/video, Claude/Codex/pi CLIs
(subprocess), a WhatsApp bot (`wa-bot` extra), an evaluation harness, and image
gen/upscale. Public surface = `merceka_core/__init__.py` (`__all__` + PEP-562 lazy
exports). `tests/contracts/` exists specifically to fail loudly if a consumer-facing
shape changes — so every public-API card here carries a "consumers still import clean"
gate.

The repo was **heavily and cleanly refactored on 2026-07-05** (dropped nbdev, split
`llm.py` into messages/retry/llm_gemini, unified CLI providers into `_cli.py`, extracted
`_select_backend()` as a truth-table). Health findings are therefore subtle — mostly
stragglers the big refactor missed, plus subprocess/lock robustness and a security gap.

## Method
Phase 1 orient (README/AGENTS/docs/plans/solutions). Phase 2 four parallel health
subagents (bugs / architecture / robustness / simplification, sonnet, file:line
evidence). Phase 3 three blue-sky ideation subagents (pain+capability / automation+
leverage / assumption-breaking+scale). Phase 4 two adversarial skeptics (cost-vs-value
at solo-4-consumer scale; repo-grounded verification). Conductor rendered verdicts and
verified the load-bearing claims directly (chat() ladder, gpu.py thread leak, `_chat_one`
dead, `build/`+`dist/` already gitignored, litellm-only-for-suppress, contract baseline).

## Survivors → cards (11)

### Health (01–08, direct-to-work briefs)
1. **01 fix: `chat()` bypasses `_select_backend()`** (`llm.py:245-269`) — no `use_codex`
   branch → `codex/` chat misroutes to Ollama; tools path ignores claude/codex native
   CLI tools. The straggler the 2026-07-05 dispatch refactor missed. **P1, confirmed.**
2. **02 robustness: `gpu_lock` timeout leaks a shared-executor thread** (`resources/gpu.py:65-93`)
   — blocking `flock` on `run_in_executor(None, …)`; on timeout the thread stays parked
   forever, `os.close(fd)` doesn't wake it; enough timeouts exhaust the default pool and
   hang the whole consuming process. Fix: non-blocking `LOCK_NB` + `asyncio.sleep` poll.
3. **03 robustness: subprocess stdin written before try/finally + streaming has no
   timeout** (`agents/{claude_code,codex,pi}.py`, `llm.py:_claude_stream` 779-820) —
   `BrokenPipeError` leaks the child + fds; `_claude_stream` never `terminate()`s and
   `wait()`s unbounded, hanging the caller's coroutine.
4. **04 robustness/security: wa_bot webhook has no `X-Hub-Signature-256` and no body cap**
   (`wa_bot/webhook.py` ~239 TODO, ~270-315) — unauthenticated public endpoint that
   triggers LLM spend + trivial memory DoS.
5. **05 simplify: remove dead litellm dep + `_chat_one`** (`llm.py:66-70`, `77-78`;
   `pyproject.toml:9`) — litellm is a heavy *required* dep imported only to run
   `suppress_debug_info=True`; cloud calls use httpx. `_chat_one` has zero callers.
6. **06 simplify: extract the 3 agent providers' identical subprocess helpers**
   (`_terminate_process`/`_close_pipe`/`_raw_events_from_stdout`, ~250 dup lines) into a
   flat `agents/_process.py` — no base class. **Depends_on 03.**
7. **07 robustness: wa_bot client collapses transient and terminal HTTP failures to
   `None`** (`wa_bot/client.py`) — silent reply loss on a 429/5xx blip; distinguish
   transient/terminal, guard `resp.json()`. No retry framework.
8. **08 refactor(doc): README omits `image.py` + `evaluation.py`** — two substantial
   public modules undocumented; add `wa_bot`-style sections. Doc-only.

### Blue-sky (09–11, re-verify at pickup)
9. **09 idea: usage/cost accounting on every response** (~85%) — OpenRouter `usage`
   already returned and discarded; attach it (side-channel, don't break return type).
10. **10 idea: `merceka doctor` CLI** (~85%) — one flat command: env keys + Ollama/CLI
    reachability + gpu_lock writable. No CLI framework.
11. **11 idea: consolidate scattered `removeprefix` model-string parsing into
    `parse_model_ref`** (~65%) — behavior-preserving; leave `_select_backend()` alone.

## Rejections (one line each)
| Candidate | Verdict | Reason |
|-----------|---------|--------|
| `VideoNotFoundError` inherits `FileNotFoundError` "footgun" | REJECT | **Intentional & contract-tested** — `errors.py:11-13` + `test_slab_surface.py:31-33` document the inheritance as load-bearing for the fallback cascade. Not a bug. |
| Golden-prompt regression harness on evaluation.py (B3) | KILL | Enterprise QA machinery for a personal lib with no SLA/team; golden prompts rot faster than used at N=1 maintainer. |
| Public-API snapshot test + deprecation-warning lane (B4) | KILL | Semver-discipline tooling for independently-versioned consumers; here a break is caught by "the 4 repos fail to import" within a day. Partial safety net already in `test_top_level_api.py`. |
| Opt-in response caching (B5) | KILL (defer) | Real cache-invalidation cost for speculative benefit; revisit only if 09's usage data shows repeated-identical spend. |
| Model alias registry (B6) | KILL | Unconfirmed demand at 4 consumers; grep-replace across 4 repos isn't costly yet. 11 covers the parsing half. |
| Gemini upload orphaned on SIGKILL (H10) | KILL | TTL-bounded, self-healing quota leak; not worth crash-recovery machinery for personal tooling. |
| `gpu_lock` → generic `resource_lock(name)` | KILL | No second lock-file need confirmed in any consumer; premature (YAGNI at N=1). |
| CLI-provider retry/backoff, fallback-chain policy, observability callback, response-normalization layer | KILL/defer | Lower confidence (≤55%); no observed pain; single-hop fallback is "enough" at this outage frequency. |
| `build/`+`dist/` "stale duplicate checked in" | REJECT | Already gitignored and untracked — a non-issue. |

## Notes for the implementer
- Re-stamp baselines and line numbers at pickup — they drift.
- The contract gate is the safety rail for this shared lib; keep `tests/contracts/`
  green and run the import smoke on every public-API card.
- `06 Depends_on 03` (same three files).
