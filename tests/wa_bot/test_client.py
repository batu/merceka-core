"""Tests for WhatsAppClient.

Tests:
- send_text(): Success, API errors, network errors, missing credentials
- send_template(): Success with components
- close(): Resource cleanup

Uses respx to mock httpx requests without making real network calls.
"""

import pytest
import httpx
import respx

from merceka_core.wa_bot import WhatsAppClient, WhatsAppConfig


@pytest.fixture
def client(sample_config: WhatsAppConfig) -> WhatsAppClient:
    """Create a WhatsAppClient with sample config."""
    return WhatsAppClient(sample_config)


@pytest.fixture
def messages_url(sample_config: WhatsAppConfig) -> str:
    """Build the expected messages API URL."""
    return (
        f"https://graph.facebook.com/"
        f"{sample_config.graph_version}/"
        f"{sample_config.phone_number_id}/messages"
    )


class TestSendText:
    """Tests for the send_text() method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_text_success(self, client: WhatsAppClient, messages_url: str):
        """Successful send should return the API response dict."""
        # Mock the WhatsApp API response
        mock_response = {
            "messaging_product": "whatsapp",
            "contacts": [{"input": "905551234567", "wa_id": "905551234567"}],
            "messages": [{"id": "wamid.HBgLMTIzNDU2Nzg5MBU="}],
        }
        respx.post(messages_url).mock(return_value=httpx.Response(200, json=mock_response))
        
        result = await client.send_text("905551234567", "Hello!")
        
        assert result is not None
        assert result["messaging_product"] == "whatsapp"
        assert result["messages"][0]["id"] == "wamid.HBgLMTIzNDU2Nzg5MBU="

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_text_api_error(self, client: WhatsAppClient, messages_url: str, capsys):
        """API errors (4xx/5xx) should return None and log a warning."""
        error_response = {
            "error": {
                "message": "Invalid OAuth access token",
                "type": "OAuthException",
                "code": 190,
            }
        }
        respx.post(messages_url).mock(return_value=httpx.Response(401, json=error_response))
        
        result = await client.send_text("905551234567", "Hello!")
        
        assert result is None
        
        # Check that a warning was logged
        captured = capsys.readouterr()
        assert "WARN" in captured.out or "failed" in captured.out.lower()

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_text_network_error(self, client: WhatsAppClient, messages_url: str, capsys):
        """Network errors should return None and log a warning."""
        # Simulate a connection error
        respx.post(messages_url).mock(side_effect=httpx.ConnectError("Connection refused"))
        
        result = await client.send_text("905551234567", "Hello!")
        
        assert result is None
        
        captured = capsys.readouterr()
        assert "WARN" in captured.out or "error" in captured.out.lower()

    @pytest.mark.asyncio
    async def test_send_text_missing_credentials(self, capsys):
        """Missing credentials should return None immediately without making a request."""
        # Create client with missing token
        config = WhatsAppConfig(
            phone_number_id="123",
            whatsapp_token="",  # Empty!
            verify_token="verify",
            waba_id="waba",
        )
        client = WhatsAppClient(config)
        
        result = await client.send_text("905551234567", "Hello!")
        
        assert result is None
        
        captured = capsys.readouterr()
        assert "Missing" in captured.out or "WARN" in captured.out

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_text_payload_format(self, client: WhatsAppClient, messages_url: str):
        """Verify the request payload format matches WhatsApp API spec."""
        respx.post(messages_url).mock(return_value=httpx.Response(200, json={"messages": []}))
        
        await client.send_text("905551234567", "Test message")
        
        # Get the request that was made
        assert len(respx.calls) == 1
        request = respx.calls[0].request
        
        # Verify headers
        assert "Bearer" in request.headers["authorization"]
        assert request.headers["content-type"] == "application/json"
        
        # Verify payload structure
        import json
        payload = json.loads(request.content)
        assert payload["messaging_product"] == "whatsapp"
        assert payload["to"] == "905551234567"
        assert payload["type"] == "text"
        assert payload["text"]["body"] == "Test message"


class TestSendTemplate:
    """Tests for the send_template() method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_template_success(self, client: WhatsAppClient, messages_url: str):
        """Successful template send should return the API response."""
        mock_response = {
            "messaging_product": "whatsapp",
            "contacts": [{"wa_id": "905551234567"}],
            "messages": [{"id": "wamid.template123"}],
        }
        respx.post(messages_url).mock(return_value=httpx.Response(200, json=mock_response))
        
        result = await client.send_template(
            to="905551234567",
            template="hello_world",
            language="en",
        )
        
        assert result is not None
        assert result["messages"][0]["id"] == "wamid.template123"

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_template_with_components(self, client: WhatsAppClient, messages_url: str):
        """Template with components should include them in the payload."""
        respx.post(messages_url).mock(return_value=httpx.Response(200, json={"messages": []}))
        
        components = [
            {
                "type": "body",
                "parameters": [
                    {"type": "text", "text": "John"},
                    {"type": "text", "text": "ORD-12345"},
                ],
            }
        ]
        
        await client.send_template(
            to="905551234567",
            template="order_ready",
            language="en",
            components=components,
        )
        
        # Verify the payload
        import json
        request = respx.calls[0].request
        payload = json.loads(request.content)
        
        assert payload["type"] == "template"
        assert payload["template"]["name"] == "order_ready"
        assert payload["template"]["language"]["code"] == "en"
        assert payload["template"]["components"] == components


class TestGetMediaUrl:
    """Tests for the get_media_url() method."""

    @pytest.fixture
    def media_url(self, sample_config: WhatsAppConfig) -> str:
        """Build the expected media API URL."""
        return f"https://graph.facebook.com/{sample_config.graph_version}/media123"

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_media_url_success(self, client: WhatsAppClient, sample_config: WhatsAppConfig):
        """Successful get should return the download URL."""
        media_id = "media123"
        api_url = f"https://graph.facebook.com/{sample_config.graph_version}/{media_id}"
        download_url = "https://lookaside.fbsbx.com/whatsapp_business/attachments/..."
        
        respx.get(api_url).mock(return_value=httpx.Response(200, json={"url": download_url}))
        
        result = await client.get_media_url(media_id)
        
        assert result == download_url

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_media_url_api_error(self, client: WhatsAppClient, sample_config: WhatsAppConfig, capsys):
        """API errors should return None."""
        media_id = "bad_media"
        api_url = f"https://graph.facebook.com/{sample_config.graph_version}/{media_id}"
        
        respx.get(api_url).mock(return_value=httpx.Response(404, json={"error": "Not found"}))
        
        result = await client.get_media_url(media_id)
        
        assert result is None
        captured = capsys.readouterr()
        assert "WARN" in captured.out

    @pytest.mark.asyncio
    async def test_get_media_url_missing_credentials(self, capsys):
        """Missing credentials should return None."""
        config = WhatsAppConfig(
            phone_number_id="123",
            whatsapp_token="",  # Empty!
            verify_token="verify",
            waba_id="waba",
        )
        client = WhatsAppClient(config)
        
        result = await client.get_media_url("media123")
        
        assert result is None
        captured = capsys.readouterr()
        assert "Missing" in captured.out or "WARN" in captured.out


class TestDownloadMedia:
    """Tests for the download_media() method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_download_media_success(self, client: WhatsAppClient, sample_config: WhatsAppConfig):
        """Successful download should return bytes."""
        media_id = "media123"
        api_url = f"https://graph.facebook.com/{sample_config.graph_version}/{media_id}"
        download_url = "https://lookaside.fbsbx.com/whatsapp_business/attachments/test.jpg"
        image_bytes = b"\x89PNG\r\n\x1a\n..."
        
        # Mock the get_media_url call
        respx.get(api_url).mock(return_value=httpx.Response(200, json={"url": download_url}))
        # Mock the actual download
        respx.get(download_url).mock(return_value=httpx.Response(200, content=image_bytes))
        
        result = await client.download_media(media_id)
        
        assert result == image_bytes

    @pytest.mark.asyncio
    @respx.mock
    async def test_download_media_url_fails(self, client: WhatsAppClient, sample_config: WhatsAppConfig):
        """If get_media_url fails, download should return None."""
        media_id = "bad_media"
        api_url = f"https://graph.facebook.com/{sample_config.graph_version}/{media_id}"
        
        respx.get(api_url).mock(return_value=httpx.Response(404, json={"error": "Not found"}))
        
        result = await client.download_media(media_id)
        
        assert result is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_download_media_download_fails(self, client: WhatsAppClient, sample_config: WhatsAppConfig):
        """If download fails, should return None."""
        media_id = "media123"
        api_url = f"https://graph.facebook.com/{sample_config.graph_version}/{media_id}"
        download_url = "https://lookaside.fbsbx.com/whatsapp_business/attachments/expired.jpg"
        
        respx.get(api_url).mock(return_value=httpx.Response(200, json={"url": download_url}))
        respx.get(download_url).mock(return_value=httpx.Response(403, text="Expired"))
        
        result = await client.download_media(media_id)
        
        assert result is None


class TestClientClose:
    """Tests for the close() method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_close_releases_resources(self, client: WhatsAppClient, messages_url: str):
        """close() should release the HTTP client."""
        respx.post(messages_url).mock(return_value=httpx.Response(200, json={}))
        
        # Make a request to initialize the client
        await client.send_text("123", "test")
        assert client._client is not None
        
        # Close should set _client to None
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_safe_to_call_twice(self, client: WhatsAppClient):
        """close() should be safe to call multiple times."""
        # Client not yet initialized
        await client.close()
        assert client._client is None
        
        # Call again - should not raise
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_close_after_use(self, client: WhatsAppClient, messages_url: str):
        """close() after use should work without errors."""
        respx.post(messages_url).mock(return_value=httpx.Response(200, json={}))
        
        await client.send_text("123", "test")
        await client.close()
        await client.close()  # Second call should be fine
        
        # Client can be reused after close (lazy reinitialization)
        respx.post(messages_url).mock(return_value=httpx.Response(200, json={}))
        result = await client.send_text("123", "test again")
        assert result is not None

