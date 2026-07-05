"""Tests for evaluation.py: calling-convention detection, config naming,
dataclass round-trips, results slicing/aggregation, and run_experiment.

No LLMs — runners are plain functions, evaluators are tiny stubs
satisfying the Evaluator protocol, save paths point at tmp_path.
"""

import json
from dataclasses import dataclass

import pytest

from merceka_core.evaluation import (
  Evaluation,
  ExperimentResults,
  TaskResult,
  _detect_calling_convention,
  config_name,
  run_experiment,
)


@dataclass
class StubTask:
  name: str
  prompt: str = ""


class StubEvaluator:
  name = "success_checker"

  def evaluate(self, output, task=None, config=None) -> list[Evaluation]:
    return [Evaluation("success", bool(output))]


def _result(task="t1", config="c1", error=None, success=None, output="out",
            duration=1.0):
  evaluations = [] if success is None else [Evaluation("success", success)]
  return TaskResult(
    output=output,
    duration=duration,
    task_name=task,
    config_name=config,
    evaluations=evaluations,
    error=error,
  )


# --- _detect_calling_convention ---

class TestDetectCallingConvention:
  def test_no_params_is_none(self):
    def f():
      return 1
    assert _detect_calling_convention(f, has_tasks=True, has_configs=True) == 'none'

  def test_single_param_with_tasks_is_task(self):
    def f(task):
      return task
    assert _detect_calling_convention(f, has_tasks=True, has_configs=False) == 'task'

  def test_params_with_configs_only_is_config(self):
    def f(lr):
      return lr
    assert _detect_calling_convention(f, has_tasks=False, has_configs=True) == 'config'

  def test_kwargs_only_with_configs_is_config(self):
    def f(**config):
      return config
    assert _detect_calling_convention(f, has_tasks=False, has_configs=True) == 'config'

  def test_task_plus_config_params_is_both(self):
    def f(task, lr):
      return task, lr
    assert _detect_calling_convention(f, has_tasks=True, has_configs=True) == 'both'

  def test_task_plus_kwargs_is_both(self):
    def f(task, **config):
      return task, config
    assert _detect_calling_convention(f, has_tasks=True, has_configs=True) == 'both'

  def test_params_but_neither_tasks_nor_configs_is_none(self):
    # Edge: the runner takes a param but the experiment supplies nothing
    # to fill it — detection falls through to 'none'.
    def f(x):
      return x
    assert _detect_calling_convention(f, has_tasks=False, has_configs=False) == 'none'

  def test_single_param_with_tasks_and_configs_is_task_only(self):
    # One positional param + tasks present: the param is claimed by the
    # task, and without **kwargs there is no slot left for the config.
    def f(task):
      return task
    assert _detect_calling_convention(f, has_tasks=True, has_configs=True) == 'task'


# --- config_name ---

class TestConfigName:
  def test_name_key_wins(self):
    assert config_name({"name": "small", "rank": 64}) == "small"

  def test_auto_generated_sorted_alphabetically(self):
    assert config_name({"rank": 64, "lr": 0.001}) == "lr_0.001_rank_64"

  def test_empty_config_is_default(self):
    assert config_name({}) == "default"

  def test_non_primitive_values_filtered(self):
    assert config_name({"layers": [1, 2], "cb": print}) == "default"
    assert config_name({"layers": [1, 2], "lr": 0.1}) == "lr_0.1"


# --- round-trips ---

class TestRoundTrips:
  def test_evaluation_round_trip(self):
    e = Evaluation(name="accuracy", value=0.93)
    assert Evaluation.from_dict(e.to_dict()) == e

  def test_task_result_round_trip_without_error(self):
    r = _result(success=True, output={"answer": 42})
    restored = TaskResult.from_dict(r.to_dict())
    assert restored.output == {"answer": 42}
    assert restored.task_name == "t1"
    assert restored.config_name == "c1"
    assert restored.evaluations == [Evaluation("success", True)]
    assert restored.error is None
    assert restored.trace_id is None

  def test_task_result_error_reconstructed_with_type_tag(self):
    r = _result(error=ValueError("bad input"))
    restored = TaskResult.from_dict(r.to_dict())
    assert isinstance(restored.error, Exception)
    assert str(restored.error) == "[ValueError] bad input"

  def test_experiment_results_save_load_round_trip(self, tmp_path):
    results = ExperimentResults(
      results=[_result(success=True), _result(task="t2", config="c2", success=False)],
      experiment_name="exp",
      description="round trip",
      git_hash="abc123",
    )
    save_dir = tmp_path / "run1"
    results.save(str(save_dir))

    assert results.save_path == str(save_dir)
    assert (save_dir / "experiment_results.json").exists()
    assert (save_dir / "result_summary.txt").exists()

    loaded = ExperimentResults.load(str(save_dir))
    assert loaded.experiment_name == "exp"
    assert loaded.description == "round trip"
    assert loaded.git_hash == "abc123"
    assert len(loaded) == 2
    assert loaded.task_names == {"t1", "t2"}

  def test_load_accepts_file_path(self, tmp_path):
    results = ExperimentResults(results=[_result()], experiment_name="exp")
    results.save(str(tmp_path))
    loaded = ExperimentResults.load(str(tmp_path / "experiment_results.json"))
    assert len(loaded) == 1
    assert loaded.save_path == str(tmp_path)

  def test_load_missing_file_raises(self, tmp_path):
    with pytest.raises(FileNotFoundError):
      ExperimentResults.load(str(tmp_path))


# --- slicing and aggregation ---

class TestSlicing:
  @pytest.fixture
  def results(self):
    return ExperimentResults(
      results=[
        _result(task="t1", config="c1", success=True),
        _result(task="t1", config="c2", success=False),
        _result(task="t2", config="c1", error=RuntimeError("boom")),
      ],
      experiment_name="exp",
    )

  def test_by_task(self, results):
    subset = results.by_task("t1")
    assert len(subset) == 2
    assert subset.task_names == {"t1"}
    assert subset.experiment_name == "exp"

  def test_by_config(self, results):
    subset = results.by_config("c1")
    assert len(subset) == 2
    assert subset.config_names == {"c1"}

  def test_getitem_int_and_string(self, results):
    assert results[0].task_name == "t1"
    assert len(results["t2"]) == 1
    assert len(results["c2"]) == 1

  def test_getitem_ambiguous_name_raises(self):
    results = ExperimentResults(
      results=[_result(task="x", config="y"), _result(task="y", config="x")],
      experiment_name="exp",
    )
    with pytest.raises(KeyError, match="ambiguous"):
      results["x"]

  def test_getitem_unknown_name_raises(self, results):
    with pytest.raises(KeyError, match="No task or config named 'nope'"):
      results["nope"]

  def test_failures_and_successes(self, results):
    assert len(results.failures) == 1
    assert results.failures[0].task_name == "t2"
    assert len(results.successes) == 2

  def test_or_combines(self, results):
    combined = results | results.by_task("t1")
    assert len(combined) == 5
    assert (results | None) is results


class TestSuccessRate:
  def test_all_bool_values_counted(self):
    results = ExperimentResults(
      results=[_result(success=True), _result(success=True), _result(success=False)],
      experiment_name="exp",
    )
    assert results.success_rate == pytest.approx(200 / 3)

  def test_non_bool_values_ignored(self):
    results = ExperimentResults(
      results=[_result(success=True), _result(success="yes"), _result(success=0.5)],
      experiment_name="exp",
    )
    assert results.success_rate == 100.0

  def test_only_non_bool_values_is_none(self):
    results = ExperimentResults(
      results=[_result(success="yes")], experiment_name="exp",
    )
    assert results.success_rate is None

  def test_no_success_evaluations_is_none(self):
    results = ExperimentResults(results=[_result()], experiment_name="exp")
    assert results.success_rate is None

  def test_empty_results_is_none(self):
    assert ExperimentResults(results=[], experiment_name="exp").success_rate is None


# --- run_experiment ---

class TestRunExperiment:
  def test_happy_path_tasks_x_configs(self, tmp_path):
    calls = []

    def runner(task, **config):
      calls.append((task.name, dict(config)))
      return f"{task.name}:{config['lr']}"

    save_dir = tmp_path / "exp"
    results = run_experiment(
      name="grid",
      run=runner,
      tasks=[StubTask("t1"), StubTask("t2")],
      configs=[{"lr": 0.1}, {"lr": 0.01}],
      evaluators=[StubEvaluator()],
      save_path=str(save_dir),
    )

    assert len(results) == 4
    assert calls == [
      ("t1", {"lr": 0.1}), ("t1", {"lr": 0.01}),
      ("t2", {"lr": 0.1}), ("t2", {"lr": 0.01}),
    ]
    assert results.task_names == {"t1", "t2"}
    assert results.config_names == {"lr_0.1", "lr_0.01"}
    assert results.success_rate == 100.0
    assert len(results.failures) == 0

    saved = json.loads((save_dir / "experiment_results.json").read_text())
    assert saved["experiment_name"] == "grid"
    assert len(saved["results"]) == 4

  def test_save_false_writes_nothing(self, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = run_experiment(name="nosave", run=lambda: "ok", save=False)
    assert len(results) == 1
    assert results.save_path is None
    assert not (tmp_path / "experiment_results").exists()

  def test_runner_exception_captured_and_evaluators_skipped(self, tmp_path):
    class ExplodingEvaluator:
      name = "never_called"

      def evaluate(self, output, task=None, config=None):
        raise AssertionError("evaluator must not run on failed runs")

    def runner():
      raise RuntimeError("boom")

    with pytest.warns(UserWarning, match="Run failed"):
      results = run_experiment(
        name="failing", run=runner, evaluators=[ExplodingEvaluator()],
        save_path=str(tmp_path / "failing"),
      )

    assert len(results.failures) == 1
    result = results[0]
    assert isinstance(result.error, RuntimeError)
    assert result.evaluations == []

  def test_name_key_stripped_from_config_kwargs(self, tmp_path):
    seen = []

    def runner(**config):
      seen.append(config)
      return "ok"

    results = run_experiment(
      name="named",
      run=runner,
      configs=[{"name": "small", "lr": 0.1}],
      save_path=str(tmp_path / "named"),
    )
    assert seen == [{"lr": 0.1}]
    assert results.config_names == {"small"}

  def test_duplicate_config_names_rejected_upfront(self):
    with pytest.raises(ValueError, match="Duplicate config name"):
      run_experiment(
        name="dupes", run=lambda: "ok", save=False,
        configs=[{"name": "same"}, {"name": "same"}],
      )

  def test_repetitions_multiply_runs(self, tmp_path):
    results = run_experiment(
      name="reps", run=lambda: "ok", repetitions=3,
      save_path=str(tmp_path / "reps"),
    )
    assert len(results) == 3
