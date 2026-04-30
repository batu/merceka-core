from importlib.metadata import metadata, requires, version

import merceka_core


def test_package_version_matches_runtime_version():
  assert version("merceka-core") == merceka_core.__version__


def test_heavy_tooling_is_optional_dependency_metadata():
  package_metadata = metadata("merceka-core")
  extras = set(package_metadata.get_all("Provides-Extra") or [])
  requirements = requires("merceka-core") or []

  assert {"notebooks", "wa-bot"}.issubset(extras)
  assert any("jupyterlab" in req and 'extra == "notebooks"' in req for req in requirements)
  assert any("python-fasthtml" in req and 'extra == "wa-bot"' in req for req in requirements)
