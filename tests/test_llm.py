

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

    monkeypatch.setattr("merceka_core.llm.subprocess.run", fake_run)
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

    monkeypatch.setattr("merceka_core.llm.subprocess.run", fake_run)
    assert LLM("codex/default").generate("x") == "ok"
