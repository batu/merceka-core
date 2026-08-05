# idea: `merceka doctor` diagnostic — one command that verifies the environment

> Blue-sky card — brainstorm/plan before implementing; re-verify repo facts at pickup
> (baselines and line numbers drift). Confidence ~85%, complexity S.

## Summary
A single diagnostic entry point that checks: required env vars per provider
(`OPENROUTER_API_KEY`, `GOOGLE_API_KEY`, and whatever image.py/`FAL_KEY`-style keys
are needed), Ollama reachability (`/api/tags`), Claude/Codex CLI availability on PATH,
and that the `gpu_lock` path (`~/.local/state/utolye/gpu.lock`) is writable and not
wedged. Reports per-provider OK/missing in one glance. There is currently **no
user-facing CLI** (`merceka_core/_cli.py` is an internal command-builder helper for the
agent providers, not an entry point) and no healthcheck code — a clean, non-duplicating
gap.

## Why it matters (this scale)
The dev context-switches across ~4 repos; every new machine/repo setup or "why did
this call fail at 2am" moment currently means manually re-deriving which key is missing
or which daemon is down. `uv run merceka doctor` amortizes that across every consumer
repo at once — the definition of a small addition that compounds.

## Watch-outs
- Keep it a **flat script / single command**, not a CLI framework with subcommands and
  plugins (adversarial review flagged framework-creep as the risk). Consider a
  `[project.scripts]` entry in pyproject or a `python -m merceka_core.doctor`.
- Read-only checks only — never write creds, never mutate the lock. For gpu_lock,
  test writability without holding the lock long enough to disturb a live GPU job.
- Don't hardcode a provider list that drifts; derive from what `LLM`/image.py actually
  read.

## Complexity / confidence
S · ~85%.
