"""Configuration management for WhatsApp bots with MODE-based environment switching.

This module provides a clean way to manage WhatsApp API credentials across
different environments (test/prod). Set the MODE environment variable to
switch between TEST_* and PROD_* prefixed variables.

Example .env file:
    MODE=test
    VERIFY_TOKEN=my_secret_token
    WHATSAPP_TOKEN=EAAG...
    TEST_PHONE_NUMBER_ID=123456789
    TEST_WABA_ID=987654321
    PROD_PHONE_NUMBER_ID=111111111
    PROD_WABA_ID=222222222

Usage:
    from merceka_core.wa_bot import get_config
    
    config = get_config()  # Automatically uses MODE to select TEST_* or PROD_*
    print(config.phone_number_id)  # "123456789" if MODE=test
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env file if present (does nothing if already loaded or file missing)
load_dotenv()


@dataclass
class WhatsAppConfig:
    """All configuration needed for WhatsApp Cloud API.
    
    Attributes:
        phone_number_id: The Phone Number ID from Meta Developer Console.
                        This identifies which WhatsApp number sends messages.
        whatsapp_token: The access token for authenticating API requests.
                       Get this from Meta Developer Console > WhatsApp > API Setup.
        verify_token: A secret string you choose for webhook verification.
                     Must match what you enter in Meta's webhook configuration.
        waba_id: WhatsApp Business Account ID. Needed for subscribing to webhooks.
        graph_version: Meta Graph API version (default: v24.0).
    """
    phone_number_id: str
    whatsapp_token: str
    verify_token: str
    waba_id: str
    graph_version: str = "v24.0"


def get_mode() -> str:
    """Get current environment mode from MODE env var.
    
    Returns:
        "test" or "prod" (defaults to "test" if MODE is not set or invalid)
    
    Example:
        >>> os.environ["MODE"] = "prod"
        >>> get_mode()
        'prod'
    """
    mode = os.getenv("MODE", "test").lower()
    
    # Validate and warn if invalid
    if mode not in ("test", "prod"):
        print(f"WARN: Invalid MODE '{mode}', defaulting to 'test'")
        return "test"
    
    return mode


def _get_prefixed_env(key: str, required: bool = True) -> str:
    """Get environment variable with MODE-based prefix.
    
    In test mode, looks for TEST_KEY first, then falls back to KEY.
    In prod mode, looks for PROD_KEY first, then falls back to KEY.
    
    Args:
        key: The base key name (e.g., "PHONE_NUMBER_ID")
        required: If True, prints a warning when the var is missing
    
    Returns:
        The environment variable value, or empty string if not found
    """
    mode = get_mode()
    prefix = "PROD_" if mode == "prod" else "TEST_"
    
    # Try prefixed version first (e.g., TEST_PHONE_NUMBER_ID)
    prefixed_key = f"{prefix}{key}"
    value = os.getenv(prefixed_key)
    
    if value:
        return value
    
    # Fall back to unprefixed version (e.g., PHONE_NUMBER_ID)
    value = os.getenv(key, "")
    
    if not value and required:
        print(f"WARN: Missing env var {prefixed_key} (or {key})")
    
    return value


def get_config() -> WhatsAppConfig:
    """Load WhatsApp configuration from environment variables.
    
    This function reads environment variables and creates a WhatsAppConfig
    object. It respects the MODE env var to select between TEST_* and PROD_*
    prefixed variables.
    
    Variables loaded:
        - VERIFY_TOKEN (shared, no prefix)
        - WHATSAPP_TOKEN (shared, no prefix)  
        - GRAPH_VERSION (shared, optional, defaults to v24.0)
        - {MODE}_PHONE_NUMBER_ID (e.g., TEST_PHONE_NUMBER_ID)
        - {MODE}_WABA_ID (e.g., TEST_WABA_ID)
    
    Returns:
        WhatsAppConfig with all credentials populated
    
    Example:
        config = get_config()
        print(f"Using phone: {config.phone_number_id}")
    """
    mode = get_mode()
    
    config = WhatsAppConfig(
        # Shared config (same for test and prod)
        verify_token=os.getenv("VERIFY_TOKEN", ""),
        whatsapp_token=os.getenv("WHATSAPP_TOKEN", ""),
        graph_version=os.getenv("GRAPH_VERSION", "v24.0"),
        # Mode-specific config
        phone_number_id=_get_prefixed_env("PHONE_NUMBER_ID"),
        waba_id=_get_prefixed_env("WABA_ID"),
    )
    
    # Log which config we're using (redact token for safety)
    phone_preview = config.phone_number_id[:6] + "..." if config.phone_number_id else "MISSING"
    print(f"CONFIG: MODE={mode}, PHONE_NUMBER_ID={phone_preview}")
    
    return config

