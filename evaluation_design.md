# Evaluation Framework Design

This document specifies the design for `merceka_core`'s experiment evaluation framework. It serves as the authoritative reference for implementation.

## Reference Implementation

The original implementation lives in Runespawn. Study these files before implementing:

- `Runespawn/runespawn/evaluation/eval_runners.py` — ExperimentRunner and TaskRunner
- `Runespawn/runespawn/evaluation/eval_datatypes.py` — TaskConfig, TaskResult, ExperimentConfig, ExperimentResults
- `Runespawn/runespawn/evaluation/issue_categorization.py` — Issue detection (evaluator pattern)
- `Runespawn/runespawn/notebooks/03_evaluation_helper.ipynb` — Usage examples

---

## Design Principles

These principles guide implementation decisions when facing unanticipated choices.

### 1. Fail Fast and Hard

No suppressed errors. No silent warnings. If something is wrong, crash immediately with a clear error message. Hidden failures cause larger problems downstream.

**Exception**: During experiment execution (not validation), individual run failures are logged and stored, but the experiment continues. This is intentional — a 6-hour experiment shouldn't abort at hour 5 because of one edge case.

### 2. YAGNI (You Aren't Gonna Need It)

Don't add features until needed. No runtime type validation. No custom analyzer protocols. No parallel execution support. These can be added later when the need arises.

### 3. Keep It Simple, Stupid (KISS)

One concept is better than two. A function is better than a class (unless the class adds clear value). Duplication is acceptable if the alternative is abstraction overhead.

### 4. Progressive Complexity (Layered Revelation)

Simple cases should be simple. Complexity appears only when you add parameters. The API should not force users to understand the full framework to run a basic experiment.

```python
# Simplest
run_experiment(name="test", run=f)

# Add comparison
run_experiment(name="test", run=f, configs=[c1, c2])

# Add tasks
run_experiment(name="test", run=f, tasks=[t1, t2], configs=[c1, c2])

# Add evaluation
run_experiment(name="test", run=f, tasks=[...], configs=[...], evaluators=[...])
```

### 5. Explicit Over Implicit

When in doubt, be explicit. No magic that users can't understand by reading the code. The one exception is the calling convention detection (see below), which is worth the magic.

### 6. Errors Must Be Helpful

Every error message should tell the user:
- What went wrong
- What was expected
- What they can do to fix it

Example:
```
ExperimentError: Failed to call train_lora(rank=64, lr=0.0002)

Your function signature: train_lora(rank, learning_rate)
Config provided: {"rank": 64, "lr": 0.0002}

Problem: Config key "lr" doesn't match any parameter.
Did you mean "learning_rate"?
```

---

## Core Abstractions

### The Conceptual Model

The framework answers: **"Which configuration generalizes best across tasks?"**

- **Task** = What you're testing on (environments, datasets, test cases)
- **Config** = What you're optimizing (hyperparameters, algorithm settings)
- **Experiment** = The process of running all task × config combinations
- **Results** = The collected outcomes for analysis

This two-dimensional structure (task × config) is intentional. It enables questions like:
- "Which config performed best across all tasks?" (generalization)
- "Which task was hardest across all configs?" (difficulty)

If you merge tasks and configs into a single concept, you lose this analytical power.

---

### Task

**What it is**: An evaluation context — an environment, dataset, or test case.

**Why it's a protocol (not a dict)**: Tasks have *identity* separate from their values. Two tasks could have identical structures but represent different things (vader dataset vs yoda dataset). The `name` field provides this identity.

**The protocol**:
```python
class Task(Protocol):
    name: str
```

That's it. Users add whatever fields they need (dataset, environment, parameters, etc.).

**Why no `num_repetitions`**: Repetition count is an experiment concern, not a task property. The task defines *what* to test; the experiment defines *how many times*.

---

### Config

**What it is**: A bag of hyperparameters to try.

**Why it's a dict (not a protocol)**: Configs have *identity equal to their values*. `{"rank": 64, "lr": 0.001}` IS the config — no separate name needed. The values define it.

**Naming**:
- If a `"name"` key is present, use it
- Otherwise, auto-generate from contents: `"rank_64_lr_0.001"`
- Duplicate configs (same values) are an error — just increase repetitions instead

**The `config_name()` function**: A free-standing function that generates a name from a config dict. Users can call it to preview what name they'd get.

```python
from merceka_core.evaluation import config_name

config_name({"rank": 64, "lr": 0.001})  # "lr_0.001_rank_64"
config_name({"name": "small", "rank": 64})  # "small"
```

---

### Experiment

**What it is**: The orchestrator that runs all task × config × repetition combinations.

**Why it's a class (not just a function)**: 
1. Experiments have identity (name) for saving/organizing
2. Many parameters — class groups them naturally
3. Can be inspected before running
4. The `run_experiment()` function is a thin wrapper for simple cases

**The class**:
```python
class Experiment:
    def __init__(
        self,
        name: str,                      # Required — for saving/organizing
        run,                            # The function to execute
        tasks: list = None,             # Optional — defaults to single dummy task
        configs: list[dict] = None,     # Optional — defaults to single empty config
        repetitions: int = 1,           # How many times per task×config
        evaluators: list = None,        # Optional — metrics to compute
        description: str = None,        # Optional — human context
    ):
        ...
    
    def run(
        self,
        save: bool = True,              # Whether to save results
        save_path: str = None,          # Custom path (auto-generated if None)
    ) -> ExperimentResults:
        ...
```

**The wrapper function**:
```python
def run_experiment(name: str, run, **kwargs) -> ExperimentResults:
    """Thin wrapper for simple cases."""
    return Experiment(name=name, run=run, **kwargs).run()
```

---

### TaskResult

**What it is**: The outcome of a single run (one task × one config × one repetition).

```python
@dataclass
class TaskResult:
    output: ...                        # What the runner returned
    duration: float                    # Seconds
    task_name: str                     # Which task
    config_name: str                   # Which config
    evaluations: list[Evaluation]      # Metrics from evaluators
    error: Exception | None            # None if successful
    trace_id: str | None               # For external tracing (optional)
```

**On failure**: `output` may be None, `evaluations` may be empty, `error` contains the exception. The run is still recorded.

---

### Evaluation

**What it is**: A named metric computed by an evaluator.

```python
@dataclass
class Evaluation:
    name: str       # e.g., "success", "perplexity", "accuracy"
    value: ...      # The metric value (any type)
```

**Duplicate names are an error**: If two evaluators produce an evaluation with the same name, fail with a clear error.

---

### ExperimentResults

**What it is**: A collection of TaskResults with aggregation and slicing capabilities.

```python
@dataclass
class ExperimentResults:
    results: list[TaskResult]
    experiment_name: str
    description: str | None
    git_hash: str
    save_path: str | None              # Where results were saved
```

**Slicing API**:
```python
results["task_name"]                   # Filter by task → ExperimentResults
results["config_name"]                 # Filter by config → ExperimentResults
results["task"]["config"]              # Chain filters

results.by_task("name")                # Explicit (for disambiguation)
results.by_config("name")              # Explicit (for disambiguation)

results[0]                             # First TaskResult
results[0].output                      # Access single run output
```

**On ambiguous key** (matches both task and config):
```
KeyError: 'gpt4o' is ambiguous — matches both task and config.
Use by_task('gpt4o') or by_config('gpt4o').
```

**On missing key**:
```
KeyError: No task or config named 'vadar'.
Available tasks: ['vader', 'yoda']
Available configs: ['rank_64', 'rank_128']
```

**Aggregation**:
```python
results.success_rate                   # Percentage (if "success" evaluation exists)
results.total_duration                 # Sum of all durations
results.get_evaluation_values("x")     # List of all values for evaluation "x"
results.failures                       # ExperimentResults with only failed runs
results.successes                      # ExperimentResults with only successful runs
```

**Combining**:
```python
combined = results_day1 | results_day2  # Merge results
```

**Persistence**:
```python
results.save(path)                     # Save to disk
ExperimentResults.load(path)           # Load from disk
```

---

## The Calling Convention

The framework detects how to call the runner based on what's provided:

| Provided | Framework calls |
|----------|-----------------|
| `run=f` only | `f()` |
| `run=f, configs=[c]` | `f(**c)` — config unpacked as kwargs |
| `run=f, tasks=[t]` | `f(t)` — task as first positional arg |
| `run=f, tasks=[t], configs=[c]` | `f(t, **c)` — both |
| `run=lambda t, c: ...` | `your_lambda(t, c)` — full control |

**This is magic, but worth it**: It enables progressive complexity without boilerplate.

**Error handling**: If the call fails (wrong signature, missing param), provide a clear error showing:
- What the framework tried to call
- The function's actual signature
- The config keys provided
- Suggestions for fixing

---

## Evaluators

**What they are**: Objects that compute metrics from run output.

**Why they exist**: 
1. Run during experiment (not after) — can capture transient state
2. Consistent storage in TaskResult.evaluations
3. Enables aggregation across runs
4. Separates "running" from "evaluating"

**The protocol**:
```python
class Evaluator(Protocol):
    name: str
    
    def evaluate(self, output, task, config: dict) -> list[Evaluation]:
        """Compute evaluations from run output."""
        ...
```

**Evaluator receives task and config**: The evaluator might need context (e.g., expected output from task, threshold from config).

**Example**:
```python
@dataclass
class SuccessEvaluator:
    name: str = "success"
    
    def evaluate(self, output, task, config) -> list[Evaluation]:
        # Success criteria might be on the task
        success = task.expected_output in str(output)
        return [Evaluation(name="success", value=success)]
```

---

## Analysis Functions (Not Analyzers)

**Why functions, not a protocol**: Analyzers run once on final results. There's no timing the framework must manage. Functions are simpler and equally powerful.

**Location**: `merceka_core.evaluation.plots`, `merceka_core.evaluation.export`, etc.

**Examples**:
```python
from merceka_core.evaluation.plots import plot_success_rates, plot_heatmap
from merceka_core.evaluation.export import to_csv

results = exp.run()
plot_success_rates(results)
plot_heatmap(results)
to_csv(results, "output.csv")
```

---

## Saving and Loading

**Default behavior**: Experiments save automatically (`save=True`).

**Rationale**: Losing a 6-hour experiment because you forgot to save is worse than disk clutter. You can always delete files; you can't recover unsaved results.

**Save location**: `./experiment_results/{name}_{timestamp}/`

**What's saved**:
- `experiment_results.json` — full data, loadable
- `result_summary.txt` — human-readable summary

**Intermediate saves**: During the experiment, results are saved after each run. If it crashes at run 15/20, runs 1-14 are preserved.

**Opt-out**: `exp.run(save=False)`

**Custom path**: `exp.run(save_path="./my/path/")`

---

## Failure Handling

**During experiment execution**: 
1. If a run fails (runner or evaluator exception), log a warning
2. Store the error in `TaskResult.error`
3. Continue to the next run
4. At the end, results include both successes and failures

**Log format**:
```
[WARN] Run 7/20 (vader × rank_64) failed: ValueError: Invalid input shape
```

**Accessing failures**:
```python
results.failures                       # ExperimentResults with failed runs
results.failures[0].error              # The exception
```

---

## API Examples

### Simplest Case
```python
from merceka_core.evaluation import run_experiment

results = run_experiment(name="sanity_check", run=train_lora)
print(results[0].output)
```

### Comparing Configs
```python
results = run_experiment(
    name="rank_comparison",
    run=train_lora,
    configs=[
        {"rank": 64, "lr": 2e-4},
        {"rank": 128, "lr": 2e-4},
        {"rank": 256, "lr": 2e-4},
    ],
)

for r in results:
    print(f"{r.config_name}: {r.output}")
```

### Full Experiment with Tasks
```python
from merceka_core.evaluation import Experiment

exp = Experiment(
    name="lora_generalization",
    description="Testing if rank 128 generalizes across characters",
    run=train_lora,
    tasks=[vader_task, yoda_task, obiwan_task],
    configs=[
        {"name": "small", "rank": 64},
        {"name": "medium", "rank": 128},
        {"name": "large", "rank": 256},
    ],
    repetitions=3,
    evaluators=[PerplexityEvaluator(), CharacterAccuracyEvaluator()],
)

results = exp.run()
```

### Analyzing Results
```python
# Slicing
results["vader"]                       # All configs on vader
results["medium"]                      # All tasks with medium config
results["vader"]["medium"]             # Specific combination

# Aggregation
print(f"Success rate: {results.success_rate}%")
print(f"Perplexities: {results.get_evaluation_values('perplexity')}")

# Visualization
from merceka_core.evaluation.plots import plot_success_rates
plot_success_rates(results)
```

### Loading Saved Results
```python
from merceka_core.evaluation import ExperimentResults

results = ExperimentResults.load("./experiment_results/lora_generalization_20251215_143052/")
print(results.success_rate)
```

### Combining Results
```python
results_v1 = ExperimentResults.load("./v1/")
results_v2 = ExperimentResults.load("./v2/")
combined = results_v1 | results_v2
```

---

## Implementation Notes

### Calling Convention Detection

Use `inspect.signature` to detect function arity. Match against what's provided (tasks, configs) to determine calling pattern.

### Config Name Generation

Sort keys alphabetically for deterministic names. Handle non-primitive values gracefully (skip them or use type name).

### Git Hash

Use `git.Repo(search_parent_directories=True).head.object.hexsha`. Catch exceptions and default to `"unknown"`.

### Intermediate Saves

Overwrite the same file after each run. Consider atomic writes (write to temp, then rename) for crash safety.

---

## What's NOT in This Design

These features were explicitly deferred (YAGNI):

- **Validation mode**: Users validate by running small experiments manually
- **Parallel execution**: Solve when needed; may require stateless evaluators
- **Analyzer protocol**: Just use functions
- **Custom config naming function**: Users preprocess configs if needed
- **`task_field` parameter**: Use lambda escape hatch instead
- **Runtime type validation**: Let it fail at call time

---

## Summary

| Abstraction | Type | Identity | Purpose |
|-------------|------|----------|---------|
| Task | Protocol (`name: str`) | Explicit name | Evaluation context (generalization) |
| Config | Dict (optional `name` key) | Values themselves | Hyperparameters to try |
| Experiment | Class | Required name | Orchestrates runs |
| TaskResult | Dataclass | task_name + config_name | Single run outcome |
| Evaluation | Dataclass | name | Single metric |
| ExperimentResults | Dataclass | experiment_name | Collection + aggregation |

