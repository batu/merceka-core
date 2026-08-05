# refactor: README omits the image.py and evaluation.py public modules

`README.md` (91 lines) documents `LLM`, `gpu_lock`, and the exception hierarchy but
never mentions two substantial, exported, user-facing modules:

- `merceka_core/image.py` (712 lines) — public functions `generate_image`,
  `edit_image`, `inpaint`, `upscale_image` (in the module's `__all__`).
- `merceka_core/evaluation.py` (790 lines) — backs the declared `evaluation`
  pyproject extra; public `Experiment`, `Task`, `Evaluator`, `ExperimentResults`,
  `run_experiment`, `to_dataframe`.

Neither appears in the README nor in `__init__.py`'s `_LAZY_EXPORTS`/`__all__`, and
neither has a contract test pinning its shape — unlike `wa_bot`, which is also
submodule-only but gets an explicit README section and Quick Start. Result: the ~4
consumer repos must read source to discover these modules exist and how to import them.

## Decided approach (doc-only, no code change)
Add two short README sections mirroring the existing `wa_bot` pattern: for each module,
a one-line purpose, the import path (`from merceka_core.image import ...` /
`from merceka_core.evaluation import ...`), and a minimal usage snippet. Note the
`evaluation` extra (`pip install "merceka-core[evaluation]"`) the way the README already
notes `wa-bot`.

Explicitly **NOT** in scope: adding root-level lazy exports for these modules (that's a
public-API surface change with consumer-import risk and no clear demand) — documenting
the existing submodule import path is cheaper and lower-risk. If a future card wants
top-level exports, that's a separate public-API change with the contract gate.

## Scope fence
- `README.md` only. No code, no `__init__.py`, no new tests.

## Acceptance criteria
- README has a section each for image generation and the evaluation harness, with a
  correct import path and a runnable-shaped snippet.
- The `evaluation` extra install line is documented.
- No code or public-API change; import surface identical.

## Verify (baseline stamped 2026-07-06, main @ 1e859de)
```
cd /home/batu/Desktop/utolye/merceka_core
uv run pytest tests/ -q   # baseline unchanged: 346 passed, 1 skipped, 6 deselected
# Sanity-check the documented import paths actually resolve:
uv run python -c "from merceka_core.image import generate_image, edit_image, inpaint, upscale_image; print('image OK')"
uv run python -c "from merceka_core.evaluation import Experiment, ExperimentResults, run_experiment; print('eval OK')"
```

## Constraints
No PRs; conductor merges. Doc-only. Out-of-fence → handoff SURPRISES.
