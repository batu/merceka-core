"""Pins for merceka_core._cli — the single home of CLI-flag knowledge."""

from merceka_core import _cli


class TestClaudeCommand:
  def test_plain_text_call_grants_no_tools(self):
    # LLM's no-tools text path: no --allowedTools, no --add-dir
    cmd = _cli.claude_command("sonnet", system_prompt="Be helpful")
    assert cmd == ["claude", "-p", "--model", "sonnet",
                   "--system-prompt", "Be helpful"]

  def test_agent_write_profile_shape(self):
    cmd = _cli.claude_command(
      "opus", system_prompt="sp", add_dirs=["/r1"],
      allowed_tools=("Read", "Edit"), stream=True, accept_edits=True)
    assert cmd[:4] == ["claude", "-p", "--model", "opus"]
    assert "--output-format" in cmd and "stream-json" in cmd
    assert ["--permission-mode", "acceptEdits"] == cmd[cmd.index("--permission-mode"):cmd.index("--permission-mode") + 2]
    assert ["--add-dir", "/r1"] == cmd[cmd.index("--add-dir"):cmd.index("--add-dir") + 2]
    assert cmd[-1] == "Read,Edit"

  def test_env_blanks_api_key(self):
    assert _cli.claude_env()["ANTHROPIC_API_KEY"] == ""


class TestCodexExecCommand:
  def test_llm_ephemeral_shape(self):
    cmd = _cli.codex_exec_command("gpt-5.2", ephemeral=True,
                                  images=["/tmp/a.jpg"])
    assert cmd[:3] == ["codex", "exec", "--ephemeral"]
    assert ["--model", "gpt-5.2"] == cmd[cmd.index("--model"):cmd.index("--model") + 2]
    assert ["-i", "/tmp/a.jpg"] == cmd[cmd.index("-i"):cmd.index("-i") + 2]
    assert cmd[-1] == "-"

  def test_default_model_uses_reasoning_effort(self):
    cmd = _cli.codex_exec_command("", reasoning_effort="high")
    assert "--model" not in cmd
    assert ["-c", 'model_reasoning_effort="high"'] == cmd[2:4]

  def test_agent_rooted_shape(self):
    cmd = _cli.codex_exec_command(
      "gpt-x", sandbox="workspace-write", cd="/root1",
      add_dirs=["/root2"], json_output=True)
    assert ["--sandbox", "workspace-write"] == cmd[cmd.index("--sandbox"):cmd.index("--sandbox") + 2]
    assert ["--cd", "/root1"] == cmd[cmd.index("--cd"):cmd.index("--cd") + 2]
    assert "--json" in cmd
    assert ["--add-dir", "/root2"] == cmd[cmd.index("--add-dir"):cmd.index("--add-dir") + 2]
    assert cmd[-1] == "-"


class TestStreamParsing:
  def test_text_delta_extracted(self):
    payload = {"type": "stream_event",
               "event": {"type": "content_block_delta",
                         "delta": {"type": "text_delta", "text": "hi"}}}
    assert _cli.claude_stream_text_delta(payload) == "hi"

  def test_non_delta_events_return_none(self):
    for payload in [{"type": "result"}, {"type": "stream_event", "event": {}},
                    {"type": "stream_event",
                     "event": {"type": "content_block_delta",
                               "delta": {"type": "input_json_delta"}}}]:
      assert _cli.claude_stream_text_delta(payload) is None

  def test_result_event_detection(self):
    assert _cli.is_claude_result_event({"type": "result"})
    assert not _cli.is_claude_result_event({"type": "stream_event"})
