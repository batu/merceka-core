---
topic: duplicated sync/async dispatch ladders diverge silently
date: 2026-07-05
affects: merceka_core/llm.py, any class with parallel sync+async code paths
problem_type: architecture_pattern
---

# Duplicated sync/async dispatch ladders diverge silently

## Problem

`LLM._generate_primary` (sync) and `LLM._agenerate_primary` (async) each held
their own copy of the 6-branch provider dispatch ladder. The copies diverged
without any test noticing:

- the async ladder never got a codex branch — `LLM("codex/...").agenerate()`
  silently routed the model name to **Ollama** (confusing model-not-found far
  from the cause);
- neither ladder had a gemini branch, so `gemini/` plain text calls also fell
  through to Ollama;
- the claude+tools cell silently **dropped the caller's tools** and ran a
  plain completion when no fallback was set — no warning, wrong answer shape.

The individual transports were well tested; the *selection* between them was
tested nowhere, because selection only existed as branch ordering inside two
big methods.

## Solution

Extract a pure decision function (`_select_backend()` returning a constant)
and make both ladders a mechanical map from decision → transport call. The
decision function:

- is directly truth-table-testable with zero transport mocking
  (`tests/test_dispatch_truth_table.py` pins prefix × tools × allowed_tools ×
  fallback for sync and async);
- raises eagerly for configurations that have no working backend, instead of
  silently misrouting (propagate-errors-not-neutral-returns);
- makes sync/async divergence structurally impossible — a new provider is one
  new constant + one entry per map, and a missing async entry is an instant
  test failure, not a silent Ollama fallthrough.

## Rules

1. When the same conditional ladder exists in a sync and an async method,
   extract the *decision* into one pure function both consume. Test the
   decision as a truth table; test the transports separately.
2. A config combination with no working backend must raise at dispatch time
   with the escape hatches named in the message — never silently drop
   arguments (tools) or misroute to a default backend.
3. When N call sites construct a "copy of self with different model"
   (fallback LLMs), one helper owns the constructor kwargs. Four hand-rolled
   copies omitted `add_dirs`/`allowed_tools`; the fifth omitted `tools`.

## See also

- PR #4, plan `docs/plans/2026-07-05-001-refactor-llm-dispatch-plan.md`
- Async streaming companion fix: producer thread + `asyncio.Queue` +
  `call_soon_threadsafe`, stop-event so early consumer exit doesn't block on
  stream exhaustion.
