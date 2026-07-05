"""Tests for evaluation.py's dependency handling: get_git_hash via subprocess
(no GitPython) and the pandas evaluation extra."""

import subprocess
from importlib.metadata import metadata, requires

from merceka_core import evaluation
from merceka_core.evaluation import get_git_hash


class TestGetGitHash:
  def test_returns_head_sha_in_a_repo(self):
    expected = subprocess.run(
      ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    assert get_git_hash() == expected
    assert len(get_git_hash()) == 40  # not "unknown" — the silent-dead bug

  def test_returns_unknown_outside_a_repo(self, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert get_git_hash() == "unknown"

  def test_returns_unknown_when_git_missing(self, monkeypatch):
    def no_git(*args, **kwargs):
      raise FileNotFoundError("git")

    monkeypatch.setattr(evaluation.subprocess, "run", no_git)
    assert get_git_hash() == "unknown"


class TestEvaluationExtra:
  def test_evaluation_extra_declares_pandas(self):
    extras = set(metadata("merceka-core").get_all("Provides-Extra") or [])
    assert "evaluation" in extras
    reqs = requires("merceka-core") or []
    assert any("pandas" in r and 'extra == "evaluation"' in r for r in reqs)

  def test_to_dataframe_error_names_the_extra(self, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def block_pandas(name, *args, **kwargs):
      if name == "pandas":
        raise ImportError("No module named 'pandas'")
      return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_pandas)
    try:
      evaluation.to_dataframe(
        evaluation.ExperimentResults(results=[], experiment_name="t"))
      raise AssertionError("expected ImportError")
    except ImportError as e:
      assert "merceka-core[evaluation]" in str(e)


class TestNoGitPython:
  def test_gitpython_not_imported(self):
    import merceka_core.evaluation as ev
    src = open(ev.__file__).read()
    assert "from git import" not in src and "import git\n" not in src
