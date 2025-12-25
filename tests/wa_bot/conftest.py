"""Shared fixtures for wa_bot tests.

This module provides common test fixtures used across multiple test files:
- sample_config: A WhatsAppConfig with test credentials
- sample_message: A Message object for testing handlers
- sample_webhook_payload: A realistic webhook payload from Meta
"""

import pytest

from merceka_core.wa_bot import WhatsAppConfig, Message


@pytest.fixture
def sample_config() -> WhatsAppConfig:
    """Create a sample WhatsAppConfig for testing.
    
    Uses fake but realistic-looking values. These should never be used
    for real API calls (tests should mock httpx).
    """
    return WhatsAppConfig(
        phone_number_id="123456789012345",
        whatsapp_token="EAAG_test_token_1234567890",
        verify_token="test_verify_secret",
        waba_id="987654321098765",
        graph_version="v24.0",
    )


@pytest.fixture
def sample_message() -> Message:
    """Create a sample Message for testing handlers."""
    return Message(
        sender="905551234567",
        text="Hello, bot!",
        message_id="wamid.HBgLMTIzNDU2Nzg5MBUCABEYEjM=",
        timestamp="1702656000",
    )


@pytest.fixture
def sample_webhook_payload() -> dict:
    """Create a realistic webhook payload as sent by Meta.
    
    This matches the structure documented in Meta's WhatsApp Cloud API docs.
    Contains a single text message from a test sender.
    """
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "987654321098765",  # WABA ID
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "123456789012345",
                            },
                            "contacts": [
                                {
                                    "profile": {"name": "Test User"},
                                    "wa_id": "905551234567",
                                }
                            ],
                            "messages": [
                                {
                                    "from": "905551234567",
                                    "id": "wamid.HBgLMTIzNDU2Nzg5MBUCABEYEjM=",
                                    "timestamp": "1702656000",
                                    "type": "text",
                                    "text": {"body": "Hello, bot!"},
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }


@pytest.fixture
def multi_message_payload() -> dict:
    """Webhook payload with multiple messages (batched delivery)."""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "987654321098765",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "123456789012345",
                            },
                            "messages": [
                                {
                                    "from": "905551234567",
                                    "id": "wamid.msg1",
                                    "timestamp": "1702656000",
                                    "type": "text",
                                    "text": {"body": "First message"},
                                },
                                {
                                    "from": "905551234567",
                                    "id": "wamid.msg2",
                                    "timestamp": "1702656001",
                                    "type": "text",
                                    "text": {"body": "Second message"},
                                },
                                {
                                    "from": "905559876543",
                                    "id": "wamid.msg3",
                                    "timestamp": "1702656002",
                                    "type": "text",
                                    "text": {"body": "Third from different user"},
                                },
                            ],
                        },
                    }
                ],
            }
        ],
    }


@pytest.fixture
def mixed_type_payload() -> dict:
    """Webhook payload with text and non-text messages."""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "987654321098765",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "123456789012345",
                            },
                            "messages": [
                                {
                                    "from": "905551234567",
                                    "id": "wamid.text1",
                                    "timestamp": "1702656000",
                                    "type": "text",
                                    "text": {"body": "Text message"},
                                },
                                {
                                    "from": "905551234567",
                                    "id": "wamid.image1",
                                    "timestamp": "1702656001",
                                    "type": "image",
                                    "image": {
                                        "mime_type": "image/jpeg",
                                        "sha256": "abc123",
                                        "id": "img123",
                                    },
                                },
                                {
                                    "from": "905551234567",
                                    "id": "wamid.audio1",
                                    "timestamp": "1702656002",
                                    "type": "audio",
                                    "audio": {
                                        "mime_type": "audio/ogg",
                                        "sha256": "def456",
                                        "id": "aud123",
                                    },
                                },
                            ],
                        },
                    }
                ],
            }
        ],
    }


@pytest.fixture
def image_with_caption_payload() -> dict:
    """Webhook payload with an image message that has a caption."""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "987654321098765",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "123456789012345",
                            },
                            "messages": [
                                {
                                    "from": "905551234567",
                                    "id": "wamid.captioned_image",
                                    "timestamp": "1702656000",
                                    "type": "image",
                                    "image": {
                                        "mime_type": "image/png",
                                        "sha256": "xyz789",
                                        "id": "img456",
                                        "caption": "Check out this event poster!",
                                    },
                                },
                            ],
                        },
                    }
                ],
            }
        ],
    }

