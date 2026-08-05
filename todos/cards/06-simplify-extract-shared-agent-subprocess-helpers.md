# simplify: extract the verbatim-duplicated subprocess helpers shared by the 3 agent providers
Depends_on: 03-robustness-subprocess-stdin-cleanup-and-stream-timeout

The three CLI agent providers duplicate identical process-lifecycle code with no
shared base (`agents/__init__.py` is just 5 lines of re-exports — the duplication is
accidental, not intentional isolation):

- `_terminate_process` (`process.terminate(); process.wait()`) is byte-for-byte
  identical: `claude_code.py:169-176`, `codex.py:186-192`, `pi.py:187-193`.
- `_close_pipe` is identical across all three.
- `_raw_events_from_stdout` (`codex.py:159-161`, `pi.py:149-151`) and
  `_raw_event_from_line` are identical except for a provider-constant substitution
  (`CODEX_PROVIDER` vs `PI_PROVIDER`).

## Decided approach (narrowed per adversarial review)
Extract **only the genuinely identical helpers** into a small plain module
`merceka_core/agents/_process.py` as module-level functions:
`terminate_process(process)`, `close_pipe(pipe)`, and parametrized
`raw_events_from_stdout(stdout, provider)` / `raw_event_from_line(line, provider)`.
Each provider imports and calls them. **No base class / mixin / inheritance** — a flat
helper module is the proportionate move at 3 call sites for a solo maintainer.

Explicitly do **NOT** unify `_text_delta_from_payload`: codex.py and pi.py have
genuinely different parsing (codex inspects multiple payload keys; pi matches
`event_type.endswith(...)` suffixes). Leave `_command()`/`_prompt`/`_text_delta_from_payload`
per-provider. Only the truly-identical lifecycle plumbing moves.

Rejected: a `SubprocessAgentProvider` Template-Method base class — over-abstraction for
3 files; harder to read than four small shared functions, and the parsing logic that
actually differs would fight the hierarchy.

## Scope fence
- New file `merceka_core/agents/_process.py`.
- `merceka_core/agents/claude_code.py`, `codex.py`, `pi.py` (replace the local copies
  with imports/calls).
Behavior must be byte-identical — pure de-duplication, zero functional change.
**Depends_on 03**: card 03 relocates the stdin write in these same three files. Land 03
first, then rebase this on top so the correctness fix and the dedup don't tangle.

## Acceptance criteria
- `_terminate_process`/`_close_pipe`/`_raw_events_from_stdout`/`_raw_event_from_line`
  exist once in `_process.py`; the three providers no longer define their own copies.
- `_text_delta_from_payload` remains per-provider (codex vs pi divergence preserved).
- All `tests/test_{claude,codex,pi}_agent_provider.py` pass unchanged — no behavior
  change.
- Consumer-import gate green (mindweaver imports `ClaudeCodeAgentProvider`/`CodexAgentProvider`).

## Verify (baseline stamped 2026-07-06, main @ 1e859de)
```
cd /home/batu/Desktop/utolye/merceka_core
uv run pytest tests/ -q            # baseline: 346 passed, 1 skipped, 6 deselected
uv run pytest tests/contracts/ -q  # baseline: 17 passed  (consumer-import gate)
uv run python -c "from merceka_core import ClaudeCodeAgentProvider, CodexAgentProvider, PiAgentProvider; print('OK')"
```

## Constraints
No PRs; conductor merges. Providers are public API — import gate mandatory. Out-of-fence
→ handoff SURPRISES.
