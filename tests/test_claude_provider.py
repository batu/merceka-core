"""Tests for the Claude CLI provider and auto-cascade fallback."""

import os
import subprocess
import pytest
from unittest.mock import patch, MagicMock

# Add merceka_core to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from merceka_core.llm import LLM, CLAUDE_CLI_TIMEOUT, create_message


# --- Provider Detection ---

class TestProviderDetection:
    def test_claude_prefix_sets_use_claude(self):
        with patch.object(LLM, '_verify'):
            llm = LLM("claude/opus", system_prompt="test")
        assert llm.use_claude is True
        assert llm.use_openrouter is False

    def test_openrouter_prefix_sets_use_openrouter(self):
        llm = LLM("openrouter/anthropic/claude-sonnet-4-6", system_prompt="test")
        assert llm.use_claude is False
        assert llm.use_openrouter is True

    def test_bare_name_is_local(self):
        with patch.object(LLM, '_verify'):
            llm = LLM("gemma4:26b", system_prompt="test")
        assert llm.use_claude is False
        assert llm.use_openrouter is False

    def test_fallback_stored(self):
        with patch.object(LLM, '_verify'):
            llm = LLM("claude/sonnet", fallback="gemma4:26b", system_prompt="test")
        assert llm.fallback == "gemma4:26b"

    def test_no_verify_for_claude(self):
        """Claude provider should not try to download from ollama."""
        with patch.object(LLM, '_verify') as mock_verify:
            LLM("claude/opus", system_prompt="test")
        mock_verify.assert_not_called()


# --- Claude CLI Call ---

class TestClaudeCall:
    def test_claude_call_builds_correct_command(self):
        with patch.object(LLM, '_verify'):
            llm = LLM("claude/sonnet", system_prompt="Be helpful")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="Hello!", stderr=""
            )
            result = llm.generate("Hi")

        mock_run.assert_called_once()
        args = mock_run.call_args
        cmd = args[0][0] if args[0] else args[1].get("args")
        assert cmd == ["claude", "-p", "--model", "sonnet", "--system-prompt", "Be helpful"]
        assert args[1]["input"] == "Hi"
        assert args[1]["env"]["ANTHROPIC_API_KEY"] == ""
        assert result == "Hello!"

    def test_claude_call_respects_timeout(self):
        with patch.object(LLM, '_verify'):
            llm = LLM("claude/opus", system_prompt="test")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            llm.generate("test", timeout=30)

        assert mock_run.call_args[1]["timeout"] == 30

    def test_claude_call_default_timeout(self):
        with patch.object(LLM, '_verify'):
            llm = LLM("claude/opus", system_prompt="test")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            llm.generate("test")

        assert mock_run.call_args[1]["timeout"] == CLAUDE_CLI_TIMEOUT

    def test_claude_call_raises_on_nonzero_exit(self):
        with patch.object(LLM, '_verify'):
            llm = LLM("claude/opus", system_prompt="test")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
            with pytest.raises(subprocess.CalledProcessError):
                llm.generate("test")


# --- Fallback ---

class TestFallback:
    def test_fallback_on_file_not_found(self):
        """When claude binary is missing, falls back to local model."""
        with patch.object(LLM, '_verify'):
            llm = LLM("claude/sonnet", fallback="gemma4:26b", system_prompt="test")

        with patch("subprocess.run", side_effect=FileNotFoundError("claude not found")):
            with patch.object(LLM, '_verify'):  # mock verify on fallback LLM too
                with patch.object(LLM, '_local_call', return_value="fallback response"):
                    result = llm.generate("hello")

        assert result == "fallback response"

    def test_fallback_on_timeout(self):
        with patch.object(LLM, '_verify'):
            llm = LLM("claude/sonnet", fallback="gemma4:26b", system_prompt="test")

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("claude", 120)):
            with patch.object(LLM, '_verify'):
                with patch.object(LLM, '_local_call', return_value="timeout fallback"):
                    result = llm.generate("hello")

        assert result == "timeout fallback"

    def test_fallback_on_called_process_error(self):
        with patch.object(LLM, '_verify'):
            llm = LLM("claude/sonnet", fallback="gemma4:26b", system_prompt="test")

        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "claude")):
            with patch.object(LLM, '_verify'):
                with patch.object(LLM, '_local_call', return_value="error fallback"):
                    result = llm.generate("hello")

        assert result == "error fallback"

    def test_no_fallback_raises(self):
        """Without fallback configured, errors propagate."""
        with patch.object(LLM, '_verify'):
            llm = LLM("claude/sonnet", system_prompt="test")

        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                llm.generate("hello")

    def test_fallback_propagates_tools(self):
        """Fallback LLM should receive the original tools."""
        def fake_tool(x: str) -> str:
            """A tool."""
            return x

        with patch.object(LLM, '_verify'):
            llm = LLM("claude/sonnet", fallback="gemma4:26b",
                       system_prompt="test", tools=[fake_tool])

        # Claude+tools should delegate to fallback immediately
        with patch.object(LLM, '_run_tool_loop', return_value=("tool result", [])) as mock_loop:
            with patch.object(LLM, '_verify'):
                result = llm.generate("hello")

        assert result == "tool result"
        mock_loop.assert_called_once()


# --- Claude + Tools Delegation ---

class TestClaudeToolsDelegation:
    def test_claude_with_tools_uses_fallback(self):
        """When Claude is primary and tools are present, delegate to fallback."""
        def search(q: str) -> str:
            """Search."""
            return "result"

        with patch.object(LLM, '_verify'):
            llm = LLM("claude/sonnet", fallback="gemma4:26b",
                       system_prompt="test", tools=[search])

        # Should use fallback's tool loop, not Claude CLI
        with patch.object(LLM, '_run_tool_loop', return_value=("searched", [])) as mock_loop:
            with patch.object(LLM, '_verify'):
                result = llm.generate("find something")

        assert result == "searched"

    def test_claude_without_tools_uses_claude(self):
        """When Claude is primary and no tools, use Claude directly."""
        with patch.object(LLM, '_verify'):
            llm = LLM("claude/sonnet", fallback="gemma4:26b", system_prompt="test")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="claude response", stderr="")
            result = llm.generate("hello")

        assert result == "claude response"


# --- Chat History ---

class TestChatHistory:
    def test_chat_excludes_system_from_history(self):
        """System prompt should not appear in the history string sent to Claude."""
        with patch.object(LLM, '_verify'):
            llm = LLM("claude/sonnet", system_prompt="You are helpful")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="reply", stderr="")
            llm.chat("Hello")

        input_text = mock_run.call_args[1]["input"]
        assert "system:" not in input_text.lower() or "You are helpful" not in input_text


# --- Async ---

class TestAsync:
    @pytest.mark.asyncio
    async def test_agenerate_with_claude(self):
        with patch.object(LLM, '_verify'):
            llm = LLM("claude/sonnet", system_prompt="test")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="async ok", stderr="")
            result = await llm.agenerate("hello")

        assert result == "async ok"

    @pytest.mark.asyncio
    async def test_agenerate_fallback(self):
        with patch.object(LLM, '_verify'):
            llm = LLM("claude/sonnet", fallback="gemma4:26b", system_prompt="test")

        with patch("subprocess.run", side_effect=FileNotFoundError):
            with patch.object(LLM, '_verify'):
                with patch.object(LLM, '_local_call', return_value="async fallback"):
                    result = await llm.agenerate("hello")

        assert result == "async fallback"
