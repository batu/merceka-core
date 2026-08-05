# simplify: remove the dead litellm dependency and the unused _chat_one helper

Two verified dead-weight items in `merceka_core/llm.py`, both safe deletions.

**A. `litellm` is a heavy REQUIRED dependency imported for one noise-suppression line.**
`llm.py:77-78`:
```python
import litellm
litellm.suppress_debug_info = True  # Stop printing "Provider List" spam
```
This is the *only* real use of litellm in the package — every other occurrence
(`__init__.py:22`, `llm_gemini.py:72`, `messages.py:39`) is the word "litellm" in a
comment/docstring describing message-format compatibility, not a call. Actual cloud
inference goes through raw `httpx` to `https://openrouter.ai/api/v1/chat/completions`
(`_cloud_call_raw`, llm.py:383-404), **not** litellm. So the library declares
`litellm>=1.80.5` as a non-optional dependency (`pyproject.toml:9`) — which pulls in
dozens of provider SDKs — purely to silence noise that only appears *because*
merceka_core imports litellm in the first place. Circular waste.

**B. `_chat_one` is dead code.** `llm.py:66-70` defines a module-level `_chat_one`
helper wrapping `ollama_chat`; it has **zero callers** anywhere in the repo (grep
across all `.py` outside `.venv`/`build` returns only the definition), and it is not
in `llm.py`'s `__all__` nor the package `_LAZY_EXPORTS`/`__all__`. Redundant with the
`LLM` class's own `_local_call`/`_local_call_raw` path.

## Decided approach
- **A:** delete `import litellm` and the `litellm.suppress_debug_info = True` line, and
  remove `"litellm>=1.80.5"` from `pyproject.toml` dependencies. Before landing, grep
  the consumer repos for a bare `import litellm` that relies on merceka_core's
  transitive pull (see gate below) — if any consumer imports litellm directly, they
  must declare it themselves; note that in the handoff rather than silently breaking
  them. If the "Provider List" spam actually resurfaces from some other transitive
  path, suppress it at its real source; do not re-add litellm as a dep.
- **B:** delete `_chat_one` (llm.py:66-70).

Rejected: keeping litellm "just in case" a future cloud path wants it — YAGNI; the
current cloud path is httpx and adding it back later is one line.

## Scope fence
- `merceka_core/llm.py` (remove the two litellm lines + `_chat_one`).
- `pyproject.toml` (drop the `litellm` dependency) and regenerate `uv.lock`
  (`uv lock`).
No other files.

## Acceptance criteria
- `grep -rn "litellm" merceka_core/` returns only comment/docstring mentions, no
  `import litellm` and no `litellm.` attribute access.
- `_chat_one` is gone; `grep -rn "_chat_one" merceka_core/ tests/` is empty.
- `uv sync` resolves without litellm; the full suite is green.
- Consumer-import gate green (below) — merceka_core still imports without litellm
  installed.

## Verify (baseline stamped 2026-07-06, main @ 1e859de)
```
cd /home/batu/Desktop/utolye/merceka_core
uv lock && uv sync
uv run pytest tests/ -q            # baseline: 346 passed, 1 skipped, 6 deselected
uv run pytest tests/contracts/ -q  # baseline: 17 passed  (consumer-import gate)
uv run python -c "import merceka_core; from merceka_core import LLM; LLM('openrouter/anthropic/claude-sonnet-4-5'); print('OK, no litellm needed')"
```
Downstream check (do before removing the dep): grep mindweaver/slab/videototext for a
bare `import litellm` relying on the transitive pull; if found, note in handoff.

## Constraints
No PRs; conductor merges. Removing a declared dependency is low-risk here but touches
the install surface consumers share — keep the import gate green. Out-of-fence →
handoff SURPRISES.
