"""Tests for wa_bot utility functions.

Tests:
- redact(): Sensitive field redaction for safe logging
- normalize_command(): Text normalization for command matching
"""

import pytest

from merceka_core.wa_bot import redact, normalize_command


class TestRedact:
    """Tests for the redact() function."""

    def test_redact_sensitive_keys(self):
        """Sensitive keys should have their values replaced with ***REDACTED***."""
        payload = {
            "token": "secret123",
            "authorization": "Bearer abc",
            "access_token": "xyz789",
            "whatsapp_token": "EAAG...",
            "api_key": "key123",
            "password": "pass123",
            "secret": "shhh",
            "user": "john",  # Not sensitive
        }
        
        result = redact(payload)
        
        # Sensitive keys should be redacted
        assert result["token"] == "***REDACTED***"
        assert result["authorization"] == "***REDACTED***"
        assert result["access_token"] == "***REDACTED***"
        assert result["whatsapp_token"] == "***REDACTED***"
        assert result["api_key"] == "***REDACTED***"
        assert result["password"] == "***REDACTED***"
        assert result["secret"] == "***REDACTED***"
        
        # Non-sensitive keys should be unchanged
        assert result["user"] == "john"

    def test_redact_nested_dict(self):
        """Nested dictionaries should be recursively processed."""
        payload = {
            "outer": "visible",
            "headers": {
                "Authorization": "Bearer token123",
                "Content-Type": "application/json",
            },
            "deep": {
                "level1": {
                    "token": "nested_secret",
                    "data": "visible_data",
                }
            }
        }
        
        result = redact(payload)
        
        assert result["outer"] == "visible"
        assert result["headers"]["Authorization"] == "***REDACTED***"
        assert result["headers"]["Content-Type"] == "application/json"
        assert result["deep"]["level1"]["token"] == "***REDACTED***"
        assert result["deep"]["level1"]["data"] == "visible_data"

    def test_redact_list(self):
        """Lists containing dicts should have dicts processed."""
        payload = {
            "items": [
                {"name": "item1", "token": "secret1"},
                {"name": "item2", "token": "secret2"},
                "plain_string",
                123,
            ]
        }
        
        result = redact(payload)
        
        assert result["items"][0]["name"] == "item1"
        assert result["items"][0]["token"] == "***REDACTED***"
        assert result["items"][1]["name"] == "item2"
        assert result["items"][1]["token"] == "***REDACTED***"
        assert result["items"][2] == "plain_string"
        assert result["items"][3] == 123

    def test_redact_case_insensitive(self):
        """Key matching should be case-insensitive."""
        payload = {
            "TOKEN": "upper",
            "Token": "title",
            "token": "lower",
            "ToKeN": "mixed",
        }
        
        result = redact(payload)
        
        assert result["TOKEN"] == "***REDACTED***"
        assert result["Token"] == "***REDACTED***"
        assert result["token"] == "***REDACTED***"
        assert result["ToKeN"] == "***REDACTED***"

    def test_redact_non_dict_passthrough(self):
        """Non-dict/list values should pass through unchanged."""
        assert redact("string") == "string"
        assert redact(123) == 123
        assert redact(None) is None
        assert redact(True) is True


class TestNormalizeCommand:
    """Tests for the normalize_command() function."""

    def test_normalize_strips_whitespace(self):
        """Leading and trailing whitespace should be removed."""
        assert normalize_command("  help  ") == "help"
        assert normalize_command("\thelp\n") == "help"
        assert normalize_command("   ") == ""

    def test_normalize_lowercases(self):
        """Text should be converted to lowercase."""
        assert normalize_command("HELP") == "help"
        assert normalize_command("Help") == "help"
        assert normalize_command("HeLp") == "help"

    def test_normalize_combined(self):
        """Stripping and lowercasing should work together."""
        assert normalize_command("  HELLO WORLD  ") == "hello world"
        assert normalize_command("\n\tSTATUS\n") == "status"

    def test_normalize_preserves_internal_spaces(self):
        """Spaces within the text should be preserved."""
        assert normalize_command("hello world") == "hello world"
        assert normalize_command("  hello   world  ") == "hello   world"

