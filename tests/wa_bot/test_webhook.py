"""Tests for webhook handling.

Tests:
- parse_webhook_payload(): Payload parsing with various message types
- create_webhook_routes(): Route creation and request handling

Uses FastHTML's test client for route testing.
"""

import json
import pytest
from unittest.mock import AsyncMock

from fasthtml.common import FastHTML
from starlette.testclient import TestClient

from merceka_core.wa_bot import (
    WhatsAppClient,
    WhatsAppConfig,
    Message,
    create_webhook_routes,
    parse_webhook_payload,
)


class TestParseWebhookPayload:
    """Tests for the parse_webhook_payload() function."""

    def test_parse_single_message(self, sample_webhook_payload: dict):
        """Should correctly extract a single text message."""
        messages = parse_webhook_payload(sample_webhook_payload)
        
        assert len(messages) == 1
        msg = messages[0]
        assert msg.sender == "905551234567"
        assert msg.text == "Hello, bot!"
        assert msg.message_id == "wamid.HBgLMTIzNDU2Nzg5MBUCABEYEjM="
        assert msg.timestamp == "1702656000"

    def test_parse_multiple_messages(self, multi_message_payload: dict):
        """Should handle batched messages from multiple users."""
        messages = parse_webhook_payload(multi_message_payload)
        
        assert len(messages) == 3
        
        # First message
        assert messages[0].sender == "905551234567"
        assert messages[0].text == "First message"
        assert messages[0].message_id == "wamid.msg1"
        
        # Second message
        assert messages[1].sender == "905551234567"
        assert messages[1].text == "Second message"
        
        # Third message from different user
        assert messages[2].sender == "905559876543"
        assert messages[2].text == "Third from different user"

    def test_parse_non_text_skipped(self, mixed_type_payload: dict):
        """Non-text messages (images, audio) should be skipped."""
        messages = parse_webhook_payload(mixed_type_payload)
        
        # Only the text message should be extracted
        assert len(messages) == 1
        assert messages[0].text == "Text message"
        assert messages[0].message_id == "wamid.text1"

    def test_parse_malformed_payload_empty(self):
        """Malformed payloads should return empty list, not crash."""
        # Empty dict
        assert parse_webhook_payload({}) == []
        
        # Missing entry
        assert parse_webhook_payload({"object": "whatsapp_business_account"}) == []
        
        # Missing changes
        assert parse_webhook_payload({
            "entry": [{"id": "123"}]
        }) == []
        
        # Missing messages
        assert parse_webhook_payload({
            "entry": [{"changes": [{"value": {}}]}]
        }) == []

    def test_parse_handles_none_gracefully(self):
        """None values in payload shouldn't crash parsing."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "from": "123",
                                        "id": "msg1",
                                        "timestamp": "12345",
                                        "type": "text",
                                        "text": None,  # Unusual but possible
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        }
        
        # Should not crash, may return message with empty text
        messages = parse_webhook_payload(payload)
        # Either empty list or message with empty text is acceptable
        assert isinstance(messages, list)

    def test_parse_missing_sender_skipped(self):
        """Messages without 'from' field should be skipped."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        # "from" is missing
                                        "id": "msg1",
                                        "timestamp": "12345",
                                        "type": "text",
                                        "text": {"body": "No sender"},
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        }
        
        messages = parse_webhook_payload(payload)
        assert len(messages) == 0


class TestCreateWebhookRoutes:
    """Tests for the create_webhook_routes() function."""

    @pytest.fixture
    def app_with_routes(self, sample_config: WhatsAppConfig):
        """Create a FastHTML app with webhook routes and a mock handler."""
        app = FastHTML()
        client = WhatsAppClient(sample_config)
        
        # Track handler calls
        handler_calls: list[tuple[WhatsAppClient, Message]] = []
        
        async def mock_handler(client: WhatsAppClient, msg: Message):
            handler_calls.append((client, msg))
        
        create_webhook_routes(
            app=app,
            client=client,
            handler=mock_handler,
            verify_token="test_verify_secret",
        )
        
        return app, handler_calls

    def test_webhook_verify_success(self, app_with_routes):
        """Valid verification request should return the challenge."""
        app, _ = app_with_routes
        
        with TestClient(app) as client:
            response = client.get(
                "/webhook",
                params={
                    "hub.mode": "subscribe",
                    "hub.verify_token": "test_verify_secret",
                    "hub.challenge": "challenge_string_12345",
                },
            )
        
        assert response.status_code == 200
        assert response.text == "challenge_string_12345"

    def test_webhook_verify_wrong_token(self, app_with_routes):
        """Wrong verify token should return 403."""
        app, _ = app_with_routes
        
        with TestClient(app) as client:
            response = client.get(
                "/webhook",
                params={
                    "hub.mode": "subscribe",
                    "hub.verify_token": "wrong_token",
                    "hub.challenge": "challenge",
                },
            )
        
        assert response.status_code == 403
        assert response.text == "forbidden"

    def test_webhook_verify_missing_params(self, app_with_routes):
        """Missing parameters should return 403."""
        app, _ = app_with_routes
        
        with TestClient(app) as client:
            # Missing hub.mode
            response = client.get(
                "/webhook",
                params={
                    "hub.verify_token": "test_verify_secret",
                    "hub.challenge": "challenge",
                },
            )
            assert response.status_code == 403
            
            # Missing hub.challenge
            response = client.get(
                "/webhook",
                params={
                    "hub.mode": "subscribe",
                    "hub.verify_token": "test_verify_secret",
                },
            )
            assert response.status_code == 403

    def test_webhook_post_calls_handler(self, app_with_routes, sample_webhook_payload: dict):
        """POST with valid payload should call the handler for each message."""
        app, handler_calls = app_with_routes
        
        with TestClient(app) as client:
            response = client.post(
                "/webhook",
                json=sample_webhook_payload,
            )
        
        assert response.status_code == 200
        assert response.text == "ok"
        
        # Handler should have been called once
        assert len(handler_calls) == 1
        _, msg = handler_calls[0]
        assert msg.sender == "905551234567"
        assert msg.text == "Hello, bot!"

    def test_webhook_post_multiple_messages(self, app_with_routes, multi_message_payload: dict):
        """POST with multiple messages should call handler for each."""
        app, handler_calls = app_with_routes
        
        with TestClient(app) as client:
            response = client.post(
                "/webhook",
                json=multi_message_payload,
            )
        
        assert response.status_code == 200
        assert len(handler_calls) == 3

    def test_webhook_post_handler_error_still_returns_200(self, sample_config: WhatsAppConfig, sample_webhook_payload: dict):
        """Handler errors should not cause 500, still return 200."""
        app = FastHTML()
        client = WhatsAppClient(sample_config)
        
        async def failing_handler(client: WhatsAppClient, msg: Message):
            raise ValueError("Handler exploded!")
        
        create_webhook_routes(
            app=app,
            client=client,
            handler=failing_handler,
            verify_token="test",
        )
        
        with TestClient(app) as test_client:
            response = test_client.post("/webhook", json=sample_webhook_payload)
        
        # Should still return 200 to prevent Meta retries
        assert response.status_code == 200
        assert response.text == "ok"

    def test_webhook_post_invalid_json(self, app_with_routes):
        """Invalid JSON should return 200 (prevent retries) but not crash."""
        app, handler_calls = app_with_routes
        
        with TestClient(app) as client:
            response = client.post(
                "/webhook",
                content="not valid json {{{",
                headers={"Content-Type": "application/json"},
            )
        
        assert response.status_code == 200
        assert len(handler_calls) == 0  # Handler not called

    def test_webhook_custom_path(self, sample_config: WhatsAppConfig):
        """Should support custom webhook path."""
        app = FastHTML()
        client = WhatsAppClient(sample_config)
        
        async def handler(client, msg):
            pass
        
        create_webhook_routes(
            app=app,
            client=client,
            handler=handler,
            verify_token="test",
            path="/custom/webhook/path",
        )
        
        with TestClient(app) as test_client:
            response = test_client.get(
                "/custom/webhook/path",
                params={
                    "hub.mode": "subscribe",
                    "hub.verify_token": "test",
                    "hub.challenge": "custom_challenge",
                },
            )
        
        assert response.status_code == 200
        assert response.text == "custom_challenge"

