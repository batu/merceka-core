"""Tests for LLM resource support (images/PDFs)."""
import base64
import tempfile
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
    """Should detect PDF MIME type from extension."""
    test_file = tmp_path / "test.pdf"
    test_file.write_bytes(b"fake pdf")
    
    result = create_message_with_resource("test", test_file)
    
    url = result["content"][1]["image_url"]["url"]
    assert url.startswith("data:application/pdf;base64,")

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

  def test_raises_for_local_model(self, tmp_path: Path):
    """Should raise error when used with local (non-openrouter) model."""
    # Note: This would try to download the model if it doesn't exist,
    # so we need to mock or skip for CI
    pytest.skip("Skipping to avoid model download in tests")
    
    test_file = tmp_path / "test.png"
    test_file.write_bytes(b"fake")
    
    llm = LLM("gemma3:27b")
    
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

