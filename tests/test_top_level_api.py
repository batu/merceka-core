"""Pins the coherent top-level API: everything downstream-critical is
importable from merceka_core, with heavy modules loaded lazily (PEP 562)."""

import subprocess
import sys

import merceka_core


class TestTopLevelExports:
  def test_flagship_and_helpers_importable_top_level(self):
    from merceka_core import (  # noqa: F401
      LLM,
      ClaudeCodeAgentProvider,
      CodexAgentProvider,
      OutputSchema,
      PiAgentProvider,
      Tool,
      create_message,
      create_message_with_resource,
      create_ollama_vision_message,
      generate_with_search_grounding,
      list_local_models,
      tool_from_callable,
    )

  def test_lazy_names_are_the_real_objects(self):
    from merceka_core.llm import LLM as direct

    assert merceka_core.LLM is direct

  def test_every_all_name_resolves(self):
    missing = [n for n in merceka_core.__all__ if not hasattr(merceka_core, n)]
    assert missing == []

  def test_dir_includes_lazy_names(self):
    assert "LLM" in dir(merceka_core)

  def test_unknown_attribute_raises_attributeerror(self):
    try:
      merceka_core.does_not_exist
      raise AssertionError("expected AttributeError")
    except AttributeError as e:
      assert "does_not_exist" in str(e)


class TestLazyImportStaysLight:
  def test_importing_package_does_not_load_llm_stack(self):
    """`import merceka_core` must not pull litellm/ollama (heavy)."""
    code = (
      "import sys; import merceka_core; "
      "heavy = [m for m in ('litellm', 'ollama', 'merceka_core.llm') "
      "if m in sys.modules]; "
      "assert not heavy, f'heavy modules loaded eagerly: {heavy}'; "
      "print('light')"
    )
    result = subprocess.run(
      [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "light"
