"""Utility functions for WhatsApp bots.

This module provides helper functions commonly needed when building bots:
- redact(): Safe logging of payloads without exposing tokens
- normalize_command(): Clean text for command matching

Usage:
    from merceka_core.wa_bot import redact, normalize_command
    
    # Safe logging
    print(redact({"token": "secret123", "user": "john"}))
    # Output: {"token": "***REDACTED***", "user": "john"}
    
    # Command matching
    if normalize_command(msg.text) == "help":
        await send_help_message()
"""

from typing import Any


def redact(obj: Any) -> Any:
    """Recursively redact sensitive fields from an object for safe logging.
    
    This function walks through dicts and lists, replacing values of
    sensitive keys with "***REDACTED***". Use this when logging webhook
    payloads or API responses that might contain tokens.
    
    Sensitive keys (case-insensitive):
    - authorization
    - token
    - access_token
    - whatsapp_token
    - bearer
    - secret
    - password
    - api_key
    
    Args:
        obj: Any Python object (dict, list, string, etc.)
    
    Returns:
        A copy of the object with sensitive values redacted.
        Non-dict/list objects are returned unchanged.
    
    Example:
        payload = {
            "from": "905551234567",
            "headers": {
                "Authorization": "Bearer EAAG1234567890",
                "Content-Type": "application/json"
            }
        }
        safe_payload = redact(payload)
        print(safe_payload)
        # {
        #     "from": "905551234567",
        #     "headers": {
        #         "Authorization": "***REDACTED***",
        #         "Content-Type": "application/json"
        #     }
        # }
    """
    # Keys that should have their values hidden
    # Using a set for O(1) lookup
    sensitive_keys = {
        "authorization",
        "token",
        "access_token",
        "whatsapp_token",
        "bearer",
        "secret",
        "password",
        "api_key",
    }
    
    if isinstance(obj, dict):
        # Process each key-value pair in the dict
        result = {}
        for key, value in obj.items():
            # Check if this key is sensitive (case-insensitive)
            if str(key).lower() in sensitive_keys:
                result[key] = "***REDACTED***"
            else:
                # Recursively process the value
                result[key] = redact(value)
        return result
    
    if isinstance(obj, list):
        # Process each item in the list
        return [redact(item) for item in obj]
    
    # For all other types (str, int, None, etc.), return as-is
    return obj


def normalize_command(text: str) -> str:
    """Normalize text for command matching.
    
    This function prepares user input for comparison with known commands:
    1. Strips leading/trailing whitespace
    2. Converts to lowercase
    
    This allows commands to be recognized regardless of how the user
    typed them: "HELP", "help", "  Help  " all become "help".
    
    Args:
        text: The raw text from the user's message
    
    Returns:
        Lowercase, stripped version of the text
    
    Example:
        # In your message handler:
        command = normalize_command(msg.text)
        
        if command == "help":
            await send_help()
        elif command == "status":
            await send_status()
        elif command in ("quit", "exit", "bye"):
            await send_goodbye()
        else:
            await handle_unknown(msg.text)  # Use original for response
    """
    return text.strip().lower()

