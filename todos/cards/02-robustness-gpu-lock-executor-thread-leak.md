# robustness: gpu_lock timeout leaks a thread from asyncio's shared default executor

`gpu_lock()` (`merceka_core/resources/gpu.py:65-93`) dispatches the blocking
`fcntl.flock(fd, LOCK_EX)` to the **shared default** executor via
`loop.run_in_executor(None, _acquire_blocking, fd)` (line 70). On
`asyncio.TimeoutError` it raises `GpuLockTimeout` and, in `finally`, does
`os.close(fd)` (line 93). But closing the fd in *this* thread does **not** wake a
different thread already parked inside the `flock()` syscall — the kernel's
`struct file` reference held by the in-flight call survives the close. The code
comment (lines 78-81) even admits "The executor task is still running and will
eventually acquire" — it treats the *lock* as handled but ignores the *thread*.

**Failure scenario (blast radius = whole process, not just the GPU caller):**
Process A holds the GPU lock and hangs (stuck WhisperX/Ollama call — the exact
scenario this lock exists for). Consumers B/C/D (mindweaver, slab, videototext)
call `gpu_lock(timeout=...)` and time out. Each timeout permanently parks one
thread in asyncio's **default** `ThreadPoolExecutor`, which is shared by every
`asyncio.to_thread`/`run_in_executor(None, …)` in the process — including
`llm.py`'s local Ollama path, `agenerate`, `astream_generate`. The default pool
caps at `min(32, cpu_count+4)`. Enough timeouts (which pile up precisely while the
holder is stuck) exhaust the pool and **all** unrelated `to_thread` work across the
consuming app silently queues forever, no error. Slow-burn, but real.

## Decided approach
Stop using a blocking executor thread that can't be cancelled. Acquire
**non-blocking** with `fcntl.flock(fd, LOCK_EX | LOCK_NB)` in a poll loop driven by
`await asyncio.sleep(<small interval>)` until success or the timeout deadline
(`loop.time()` based). On timeout, no thread is parked — just close the fd and
raise `GpuLockTimeout`. This keeps the event loop responsive without ever occupying
an executor thread, preserves the kernel-releases-on-death property (still an fd
`flock`), and the `timeout=None` case simply loops without a deadline.

Rejected alternative: a dedicated single-worker `ThreadPoolExecutor` per acquire —
still leaks the worker on timeout (the thread is stuck in the syscall); only isolates
the damage from the shared pool rather than eliminating it. The LOCK_NB poll loop
removes the leak entirely.

Watch-out: pick a poll interval that balances acquire latency vs. wakeups (e.g.
25-50 ms); document it. Preserve exact public behavior: async CM,
`timeout` kwarg default `None`, raises `GpuLockTimeout`, path unchanged.

## Scope fence
- `merceka_core/resources/gpu.py` only.
- `tests/integration/test_gpu_lock_cross_process.py` and the gpu_lock contract
  assertions in `tests/contracts/{test_slab_surface,test_mindweaver_surface}.py`
  must still pass unchanged (they pin the public shape: callable, `timeout` kwarg
  default `None`, async CM, `GpuLockTimeout` on contention).

## Acceptance criteria
- A timeout no longer leaves a live thread blocked in `flock` (verify: after N
  `gpu_lock(timeout=0.1)` calls while the lock is held elsewhere, the default
  executor's thread count / `threading.active_count()` does not grow per call).
- Contract tests for gpu_lock still green (factory, `timeout` kwarg, async CM,
  `GpuLockTimeout` raised on cross-fd contention).
- Cross-process integration test (`-m gpu`) still serializes correctly.

## Verify (baseline stamped 2026-07-06, main @ 1e859de)
```
cd /home/batu/Desktop/utolye/merceka_core
uv run pytest tests/ -q                         # baseline: 346 passed, 1 skipped, 6 deselected
uv run pytest tests/contracts/ -q               # baseline: 17 passed  (consumer-import gate)
uv run pytest -m gpu tests/integration/test_gpu_lock_cross_process.py -q   # gpu-marked; run on a box with fcntl
uv run python -c "from merceka_core import gpu_lock, GpuLockTimeout; import inspect; print('OK', 'timeout' in inspect.signature(gpu_lock).parameters)"
```

## Constraints
No PRs; conductor merges. `gpu_lock` is imported at startup by mindweaver/slab —
the contract gate is mandatory. Out-of-fence needs → handoff SURPRISES.
