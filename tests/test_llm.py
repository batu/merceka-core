import pytest


class TestCodexProvider:
  """codex/ prefix dispatches to the Codex CLI subprocess."""

  def test_codex_call_builds_command(self, monkeypatch):
    from merceka_core.llm import LLM
    captured = {}

    def fake_run(cmd, **kwargs):
      captured["cmd"] = cmd
      captured["input"] = kwargs.get("input")
      class R:
        returncode = 0
        stdout = "hello"
        stderr = ""
      return R()

    monkeypatch.setattr("merceka_core._cli.run_cli", fake_run)
    llm = LLM("codex/gpt-5.2", system_prompt="SYS")
    out = llm.generate("MSG", images=["/tmp/a.jpg"])
    assert out == "hello"
    assert captured["cmd"][:2] == ["codex", "exec"]
    idx = captured["cmd"].index("--model")
    assert ["--model", "gpt-5.2"] == captured["cmd"][idx:idx + 2]
    assert "--ephemeral" in captured["cmd"]
    assert ["-i", "/tmp/a.jpg"] == captured["cmd"][captured["cmd"].index("-i"):captured["cmd"].index("-i") + 2]
    assert captured["cmd"][-1] == "-"
    assert captured["input"].startswith("SYS\n\nMSG")

  def test_codex_default_omits_model_flag(self, monkeypatch):
    from merceka_core.llm import LLM

    def fake_run(cmd, **kwargs):
      assert "--model" not in cmd
      class R:
        returncode = 0
        stdout = "ok"
        stderr = ""
      return R()

    monkeypatch.setattr("merceka_core._cli.run_cli", fake_run)
    assert LLM("codex/default").generate("x") == "ok"


class TestVerify:
  """LLM(<ollama model>) pulls the model only when it is not installed."""

  @pytest.fixture
  def ollama(self, monkeypatch):
    import merceka_core.llm as llm_module

    state = {"installed": [], "pulled": []}
    monkeypatch.setattr(llm_module, "list_local_models", lambda: list(state["installed"]))
    monkeypatch.setattr(llm_module, "_download_model", state["pulled"].append)
    return state

  @pytest.mark.parametrize("name,installed", [
    ("gemma3", ["gemma3:latest"]),  # Ollama lists the implicit :latest tag
    ("gemma3:latest", ["gemma3:latest"]),
    ("gemma4:26b", ["gemma4:26b", "gemma3:latest"]),
    ("hf.co/org/model", ["hf.co/org/model:latest"]),
    ("localhost:5000/model", ["localhost:5000/model:latest"]),  # port, not a tag
  ])
  def test_installed_model_is_not_pulled(self, ollama, name, installed):
    """Regression: LLM("gemma3") pulled on every construction."""
    from merceka_core.llm import LLM

    ollama["installed"] = installed
    LLM(name)
    assert ollama["pulled"] == []

  @pytest.mark.parametrize("name,installed", [
    ("gemma3", ["gemma3:1b"]),
    ("gemma3:1b", ["gemma3:latest"]),
    ("gemma3", []),
  ])
  def test_missing_model_is_pulled(self, ollama, name, installed):
    from merceka_core.llm import LLM

    ollama["installed"] = installed
    LLM(name)
    assert ollama["pulled"] == [name]
