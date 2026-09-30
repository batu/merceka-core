from importlib.metadata import metadata, requires, version

import merceka_core


def test_package_version_matches_runtime_version():
  assert version("merceka-core") == merceka_core.__version__


def test_dependency_metadata_stays_lean():
  package_metadata = metadata("merceka-core")
  extras = set(package_metadata.get_all("Provides-Extra") or [])
  requirements = requires("merceka-core") or []

  # wa_bot and evaluation were removed 2026-09-30 (no importers on either machine);
  # the notebook layer was removed 2026-07-05.
  assert not extras & {"wa-bot", "evaluation", "notebooks"}
  assert not any(
    name in req
    for req in requirements
    for name in ("litellm", "python-fasthtml", "pandas", "jupyterlab", "nbdev")
  )
