"""Tests for wa_bot configuration management.

Tests:
- get_mode(): MODE environment variable handling
- get_config(): Loading config with MODE-based prefixes

These tests use monkeypatch to manipulate environment variables,
ensuring isolation between tests.
"""

import pytest


class TestGetMode:
    """Tests for the get_mode() function."""

    def test_get_mode_default(self, monkeypatch):
        """When MODE is not set, should default to 'test'."""
        # Remove MODE if it exists
        monkeypatch.delenv("MODE", raising=False)
        
        # Import fresh to pick up env changes
        # (We need to reload because config.py reads env at import time)
        from merceka_core.wa_bot.config import get_mode
        
        assert get_mode() == "test"

    def test_get_mode_test(self, monkeypatch):
        """When MODE=test, should return 'test'."""
        monkeypatch.setenv("MODE", "test")
        
        from merceka_core.wa_bot.config import get_mode
        
        assert get_mode() == "test"

    def test_get_mode_prod(self, monkeypatch):
        """When MODE=prod, should return 'prod'."""
        monkeypatch.setenv("MODE", "prod")
        
        from merceka_core.wa_bot.config import get_mode
        
        assert get_mode() == "prod"

    def test_get_mode_case_insensitive(self, monkeypatch):
        """MODE should be case-insensitive."""
        monkeypatch.setenv("MODE", "PROD")
        
        from merceka_core.wa_bot.config import get_mode
        
        assert get_mode() == "prod"
        
        monkeypatch.setenv("MODE", "Test")
        assert get_mode() == "test"

    def test_get_mode_invalid_fallback(self, monkeypatch, capsys):
        """Invalid MODE values should fall back to 'test' with a warning."""
        monkeypatch.setenv("MODE", "staging")
        
        from merceka_core.wa_bot.config import get_mode
        
        result = get_mode()
        
        assert result == "test"
        
        # Check that a warning was printed
        captured = capsys.readouterr()
        assert "Invalid MODE" in captured.out or "invalid" in captured.out.lower()


class TestGetConfig:
    """Tests for the get_config() function."""

    def test_get_config_test_mode(self, monkeypatch, capsys):
        """In test mode, should load TEST_* prefixed variables."""
        # Set up environment
        monkeypatch.setenv("MODE", "test")
        monkeypatch.setenv("VERIFY_TOKEN", "shared_verify")
        monkeypatch.setenv("WHATSAPP_TOKEN", "shared_token")
        monkeypatch.setenv("GRAPH_VERSION", "v25.0")
        monkeypatch.setenv("TEST_PHONE_NUMBER_ID", "test_phone_123")
        monkeypatch.setenv("TEST_WABA_ID", "test_waba_456")
        monkeypatch.setenv("PROD_PHONE_NUMBER_ID", "prod_phone_789")
        monkeypatch.setenv("PROD_WABA_ID", "prod_waba_012")
        
        from merceka_core.wa_bot.config import get_config
        
        config = get_config()
        
        # Shared config
        assert config.verify_token == "shared_verify"
        assert config.whatsapp_token == "shared_token"
        assert config.graph_version == "v25.0"
        
        # Should use TEST_* prefixed values
        assert config.phone_number_id == "test_phone_123"
        assert config.waba_id == "test_waba_456"

    def test_get_config_prod_mode(self, monkeypatch, capsys):
        """In prod mode, should load PROD_* prefixed variables."""
        # Set up environment
        monkeypatch.setenv("MODE", "prod")
        monkeypatch.setenv("VERIFY_TOKEN", "shared_verify")
        monkeypatch.setenv("WHATSAPP_TOKEN", "shared_token")
        monkeypatch.setenv("TEST_PHONE_NUMBER_ID", "test_phone_123")
        monkeypatch.setenv("TEST_WABA_ID", "test_waba_456")
        monkeypatch.setenv("PROD_PHONE_NUMBER_ID", "prod_phone_789")
        monkeypatch.setenv("PROD_WABA_ID", "prod_waba_012")
        
        from merceka_core.wa_bot.config import get_config
        
        config = get_config()
        
        # Should use PROD_* prefixed values
        assert config.phone_number_id == "prod_phone_789"
        assert config.waba_id == "prod_waba_012"

    def test_get_config_default_graph_version(self, monkeypatch, capsys):
        """When GRAPH_VERSION not set, should default to v24.0."""
        monkeypatch.setenv("MODE", "test")
        monkeypatch.setenv("VERIFY_TOKEN", "verify")
        monkeypatch.setenv("WHATSAPP_TOKEN", "token")
        monkeypatch.setenv("TEST_PHONE_NUMBER_ID", "phone")
        monkeypatch.setenv("TEST_WABA_ID", "waba")
        monkeypatch.delenv("GRAPH_VERSION", raising=False)
        
        from merceka_core.wa_bot.config import get_config
        
        config = get_config()
        
        assert config.graph_version == "v24.0"

    def test_get_config_missing_vars_empty_string(self, monkeypatch, capsys):
        """Missing environment variables should result in empty strings."""
        monkeypatch.setenv("MODE", "test")
        # Only set some vars, leave others missing
        monkeypatch.setenv("VERIFY_TOKEN", "verify")
        monkeypatch.delenv("WHATSAPP_TOKEN", raising=False)
        monkeypatch.delenv("TEST_PHONE_NUMBER_ID", raising=False)
        monkeypatch.delenv("TEST_WABA_ID", raising=False)
        
        from merceka_core.wa_bot.config import get_config
        
        config = get_config()
        
        assert config.verify_token == "verify"
        assert config.whatsapp_token == ""
        assert config.phone_number_id == ""
        assert config.waba_id == ""
        
        # Should have printed warnings about missing vars
        captured = capsys.readouterr()
        assert "WARN" in captured.out or "Missing" in captured.out

