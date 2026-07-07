# robustness: subprocess stdin written before try/finally + streaming Popen has no timeout/terminate

Two related subprocess-lifecycle gaps that leak child processes/fds and can hang a
consumer's coroutine indefinitely.

**A. stdin write happens before the cleanup try-block (3 CLI agent providers).**
`agents/claude_code.py:81-82`, `agents/codex.py:87-88`, `agents/pi.py:80-81` all do
`process.stdin.write(...)` + `process.stdin.close()` immediately after `Popen(...)`,
**before** the `try:` whose `finally` runs `_terminate_process`/`_close_pipe`. If the
child exits/crashes at once (auth failure, bad args) and closes its stdin end, the
synchronous `write()` raises `BrokenPipeError` outside the try, so the child is never
`terminate()`d/reaped and the pipes are never closed — leaked subprocess + fds that
accumulate under load. (Secondary, lower-confidence: the whole prompt is written in one
blocking call before any stdout/stderr draining starts; a child that emits enough
stderr early could fill its pipe buffer and deadlock. Fix A resolves the leak; note the
deadlock risk but don't build a full async-pump rewrite for it.)

**B. `LLM._claude_stream` uses raw Popen with no timeout and no terminate.**
`merceka_core/llm.py:779-820`: `subprocess.Popen(...)` with `stdin.write`/`close` at
~799-800 (before the `try` at ~802), `for line in process.stdout` with no bound, and a
`finally` that only does `stdout.close(); stderr.close(); process.wait()` — **no
`terminate()`/`kill()`, no `wait(timeout=...)`**. If the Claude CLI hangs (network
stall, never emits a result event) the loop blocks forever; via `astream_generate`'s
`await asyncio.shield(producer)` (llm.py:~880) that hang propagates into the calling
coroutine's teardown (`aclose()` can hang). This is strictly worse than the sibling
`agents/claude_code.py`, which does `terminate()` in the same spot — and worse than
`_claude_call`/`_codex_call`, which pass `timeout=` to `subprocess.run`.

## Decided approach
- **A:** move `process.stdin.write(...)` / `.close()` **inside** the existing `try`
  block in each of the three providers so a `BrokenPipeError` there hits the `finally`
  and the child is terminated + pipes closed. Guard the write with a narrow
  `except BrokenPipeError` that lets cleanup run (child already dead → surface a clear
  provider error, don't swallow to a neutral return).
- **B:** in `_claude_stream`, add `terminate()` (then `kill()` fallback) to the
  `finally`, and bound `process.wait()` with a `timeout=` + `kill()` on expiry, mirroring
  `agents/claude_code.py`'s pattern. Keep the streaming iteration but ensure abandonment
  (consumer stops early / generator `aclose`) reliably tears the child down.

Rejected: rewriting the providers to fully async stdin pumping — out of scope; the
confirmed leak is the pre-try write and the missing stream teardown, fix those.

## Scope fence
- `merceka_core/agents/claude_code.py`, `merceka_core/agents/codex.py`,
  `merceka_core/agents/pi.py` (the stdin-write relocation only).
- `merceka_core/llm.py` — `_claude_stream` teardown only.
Note: card 06 also edits these three agent files (helper extraction). Whichever lands
first, the other rebases. **Depends_on ordering: land 03 before 06** so the correctness
fix isn't tangled with the dedup move.

## Acceptance criteria
- In all three providers, an immediate child-exit during the stdin write triggers
  `_terminate_process`/`_close_pipe` (no orphaned process, no leaked fds) — add a test
  that stubs `Popen` with a process whose `stdin.write` raises `BrokenPipeError` and
  asserts cleanup ran.
- `_claude_stream`'s `finally` calls `terminate()` and bounds `wait()`; a hung/abandoned
  stream does not leave a live child (test via a fake Popen that never EOFs).
- Provider streaming behavior for the happy path is unchanged (existing
  `tests/test_*_agent_provider.py` stay green).

## Verify (baseline stamped 2026-07-06, main @ 1e859de)
```
cd /home/batu/Desktop/utolye/merceka_core
uv run pytest tests/ -q            # baseline: 346 passed, 1 skipped, 6 deselected
uv run pytest tests/contracts/ -q  # baseline: 17 passed  (consumer-import gate)
uv run python -c "from merceka_core import ClaudeCodeAgentProvider, CodexAgentProvider, PiAgentProvider; print('OK')"
```

## Constraints
No PRs; conductor merges. Agent providers are public (mindweaver imports them) —
contract + import gate mandatory. Out-of-fence needs → handoff SURPRISES.
Depends_on: land before 06.
