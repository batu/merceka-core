"""Tests for LLM resource support (images/PDFs)."""
import base64
from pathlib import Path

import pytest

from merceka_core.llm import create_message_with_resource, LLM


class TestCreateMessageWithResource:
  """Tests for create_message_with_resource function."""

  def test_creates_correct_structure(self, tmp_path: Path):
    """Should create message with text and image_url parts."""
    # Create a small test file
    test_file = tmp_path / "test.png"
    test_file.write_bytes(b"fake png data")
    
    result = create_message_with_resource("What's in this image?", test_file)
    
    assert result["role"] == "user"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 2
    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["text"] == "What's in this image?"
    assert result["content"][1]["type"] == "image_url"
    assert "image_url" in result["content"][1]

  def test_encodes_file_as_base64(self, tmp_path: Path):
    """Should base64 encode the file contents."""
    test_file = tmp_path / "test.txt"
    test_content = b"hello world"
    test_file.write_bytes(test_content)
    
    result = create_message_with_resource("test", test_file)
    
    url = result["content"][1]["image_url"]["url"]
    # Extract base64 part after "data:...;base64,"
    base64_data = url.split(",")[1]
    decoded = base64.b64decode(base64_data)
    assert decoded == test_content

  def test_detects_png_mime_type(self, tmp_path: Path):
    """Should detect PNG MIME type from extension."""
    test_file = tmp_path / "test.png"
    test_file.write_bytes(b"fake")
    
    result = create_message_with_resource("test", test_file)
    
    url = result["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")

  def test_detects_pdf_mime_type(self, tmp_path: Path):
    """PDFs go as an OpenRouter file part, not as an image_url.

    Regression: a PDF was sent as image_url. OpenRouter documents PDF input
    as {"type": "file", "file": {"filename", "file_data": <data URL>}}.
    """
    test_file = tmp_path / "test.pdf"
    test_file.write_bytes(b"fake pdf")

    result = create_message_with_resource("test", test_file)

    part = result["content"][1]
    assert part["type"] == "file"
    assert part["file"]["filename"] == "test.pdf"
    assert part["file"]["file_data"] == "data:application/pdf;base64," + base64.b64encode(
      b"fake pdf").decode()
    assert "image_url" not in part

  def test_detects_jpeg_mime_type(self, tmp_path: Path):
    """Should detect JPEG MIME type from extension."""
    for ext in [".jpg", ".jpeg"]:
      test_file = tmp_path / f"test{ext}"
      test_file.write_bytes(b"fake")
      
      result = create_message_with_resource("test", test_file)
      
      url = result["content"][1]["image_url"]["url"]
      assert url.startswith("data:image/jpeg;base64,")

  def test_accepts_string_path(self, tmp_path: Path):
    """Should accept string path as well as Path object."""
    test_file = tmp_path / "test.png"
    test_file.write_bytes(b"fake")
    
    result = create_message_with_resource("test", str(test_file))
    
    assert result["role"] == "user"
    assert len(result["content"]) == 2

  def test_role_parameter(self, tmp_path: Path):
    """Should respect role parameter."""
    test_file = tmp_path / "test.png"
    test_file.write_bytes(b"fake")
    
    result = create_message_with_resource("test", test_file, role="assistant")
    
    assert result["role"] == "assistant"


class TestLLMGenerateWithResource:
  """Tests for LLM.generate_with_resource method."""

  def test_codex_image_goes_to_codex_exec_with_i_flag(self, tmp_path: Path, monkeypatch):
    """Regression: codex/ models were sent to ollama_chat(model="codex/...")."""
    import merceka_core.llm as llm_module

    png = tmp_path / "shot.png"
    png.write_bytes(b"\x89PNG fake")
    seen = {}

    def fake_run(cmd, **kwargs):
      seen["cmd"], seen["input"] = cmd, kwargs["input"]

      class Result:
        returncode = 0
        stdout = "an arrow"
        stderr = ""
      return Result()

    def ollama_trap(**_kwargs):
      raise AssertionError("codex model must not reach Ollama")

    monkeypatch.setattr(llm_module.subprocess, "run", fake_run)
    monkeypatch.setattr(llm_module, "ollama_chat", ollama_trap)
    llm = LLM("codex/gpt-5", system_prompt="SYS")
    assert llm.generate_with_resource("what is this?", png) == "an arrow"
    i = seen["cmd"].index("-i")
    assert seen["cmd"][i + 1] == str(png)
    assert seen["input"] == "SYS\n\nwhat is this?"

  def test_codex_async_image_goes_to_codex_exec(self, tmp_path: Path, monkeypatch):
    import asyncio

    import merceka_core.llm as llm_module

    png = tmp_path / "shot.png"
    png.write_bytes(b"\x89PNG fake")
    seen = []

    def fake_run(cmd, **_kwargs):
      seen.append(cmd)

      class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""
      return Result()

    monkeypatch.setattr(llm_module.subprocess, "run", fake_run)
    assert asyncio.run(LLM("codex/default").agenerate_with_resource("x", png)) == "ok"
    assert str(png) in seen[0]

  def test_codex_non_image_resource_raises(self, tmp_path: Path, monkeypatch):
    """codex exec attaches images only (-i); a PDF has no route."""
    import asyncio

    import merceka_core.llm as llm_module

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(
      llm_module.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("ran codex"))
    llm = LLM("codex/gpt-5")
    with pytest.raises(ValueError, match="codex"):
      llm.generate_with_resource("x", pdf)
    with pytest.raises(ValueError, match="codex"):
      asyncio.run(llm.agenerate_with_resource("x", pdf))

  def test_raises_for_local_model(self, tmp_path: Path):
    """Should raise error when used with local (non-openrouter) model."""
    # Note: This would try to download the model if it doesn't exist,
    # so we need to mock or skip for CI
    pytest.skip("Skipping to avoid model download in tests")
    
    test_file = tmp_path / "test.png"
    test_file.write_bytes(b"fake")
    
    llm = LLM("gemma4:26b")
    
    with pytest.raises(ValueError, match="only works with cloud models"):
      llm.generate_with_resource("test", test_file)

  def test_works_with_openrouter_model(self, tmp_path: Path):
    """Should work with openrouter models."""
    test_file = tmp_path / "test.png"
    test_file.write_bytes(b"fake")
    
    llm = LLM("openrouter/google/gemini-2.5-flash")
    
    # Just verify it doesn't raise - actual API call would need mocking
    # or a live integration test
    assert llm.use_openrouter is True
    # The actual call would need an API key and would hit the network

