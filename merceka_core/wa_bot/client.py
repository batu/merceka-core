"""WhatsApp Cloud API client for sending messages.

This module provides a client class for interacting with the WhatsApp Cloud API.
It handles authentication, connection pooling, and message sending.

The client uses httpx.AsyncClient for efficient connection reuse - this means
multiple API calls will share the same underlying TCP connection, reducing
latency and resource usage.

Usage:
    from merceka_core.wa_bot import WhatsAppClient, get_config
    
    config = get_config()
    client = WhatsAppClient(config)
    
    # Send a text message
    await client.send_text("905551234567", "Hello from the bot!")
    
    # Don't forget to close when done (e.g., on app shutdown)
    await client.close()

Dependencies:
    - httpx: pip install httpx
"""

from typing import Any

import httpx

from .config import WhatsAppConfig


class WhatsAppClient:
    """Client for WhatsApp Cloud API with connection pooling.
    
    This class wraps the WhatsApp Cloud API, handling:
    - Authentication via Bearer token
    - Connection reuse for better performance
    - Error handling with None returns (no exceptions)
    
    The client creates its HTTP connection lazily (on first use) and reuses
    it for subsequent requests. Call close() when you're done to clean up.
    
    Attributes:
        config: The WhatsAppConfig containing credentials and settings.
    
    Example:
        # In a FastHTML app
        config = get_config()
        client = WhatsAppClient(config)
        
        @app.on_event("shutdown")
        async def cleanup():
            await client.close()
    """
    
    def __init__(self, config: WhatsAppConfig):
        """Initialize the client with configuration.
        
        Args:
            config: WhatsAppConfig containing phone_number_id, whatsapp_token, etc.
        
        Note:
            The HTTP client is not created here - it's created lazily on first use.
            This makes initialization lightweight and avoids issues if you create
            the client before the async event loop is running.
        """
        self.config = config
        # HTTP client created lazily on first request
        self._client: httpx.AsyncClient | None = None
    
    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client (lazy initialization).
        
        This creates the httpx.AsyncClient on first call and reuses it
        for subsequent calls. The client is configured with:
        - 15 second timeout (reasonable for WhatsApp API)
        - Connection pooling enabled by default
        
        Returns:
            The shared httpx.AsyncClient instance
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client
    
    def _get_headers(self) -> dict[str, str]:
        """Build HTTP headers for API requests.
        
        Returns:
            Dict with Authorization and Content-Type headers
        """
        return {
            "Authorization": f"Bearer {self.config.whatsapp_token}",
            "Content-Type": "application/json",
        }
    
    def _get_messages_url(self) -> str:
        """Build the messages endpoint URL.
        
        The URL format is:
        https://graph.facebook.com/{version}/{phone_number_id}/messages
        
        Returns:
            Full URL for the messages endpoint
        """
        return (
            f"https://graph.facebook.com/"
            f"{self.config.graph_version}/"
            f"{self.config.phone_number_id}/messages"
        )
    
    async def send_text(self, to: str, body: str) -> dict[str, Any] | None:
        """Send a text message to a WhatsApp user.
        
        Args:
            to: Recipient's WhatsApp ID (phone number without + or spaces).
               Example: "905551234567" for a Turkish number.
            body: The message text to send. Can include emojis and formatting
                 (use *bold*, _italic_, ~strikethrough~, ```monospace```).
        
        Returns:
            The API response as a dict if successful, containing:
            - messaging_product: "whatsapp"
            - contacts: List of recipient info
            - messages: List with message ID
            
            Returns None if the request fails (network error, invalid token, etc.)
        
        Example:
            result = await client.send_text("905551234567", "Hello! 👋")
            if result:
                print(f"Sent! Message ID: {result['messages'][0]['id']}")
            else:
                print("Failed to send")
        """
        # Check for missing credentials early
        if not self.config.whatsapp_token or not self.config.phone_number_id:
            print("WARN: Missing WHATSAPP_TOKEN or PHONE_NUMBER_ID, skipping send")
            return None
        
        # Build the request payload
        # See: https://developers.facebook.com/docs/whatsapp/cloud-api/messages/text-messages
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": body},
        }
        
        try:
            client = self._get_client()
            resp = await client.post(
                self._get_messages_url(),
                headers=self._get_headers(),
                json=payload,
            )
            
            # Check for API errors
            if resp.status_code >= 400:
                print(f"WARN: send_text failed {resp.status_code}: {resp.text}")
                return None
            
            return resp.json()
            
        except httpx.RequestError as e:
            # Network errors (timeout, connection refused, etc.)
            print(f"WARN: send_text network error: {e}")
            return None
    
    async def send_template(
        self,
        to: str,
        template: str,
        language: str = "en",
        components: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Send a template message to a WhatsApp user.
        
        Template messages are pre-approved message formats that can be sent
        outside the 24-hour messaging window. They must be created and approved
        in the Meta Business Manager before use.
        
        Args:
            to: Recipient's WhatsApp ID (phone number without + or spaces).
            template: The template name as defined in Meta Business Manager.
                     Example: "hello_world", "order_confirmation"
            language: Language code for the template (default: "en").
                     Use the code shown in Meta Business Manager.
                     Examples: "en", "en_US", "tr", "es"
            components: Optional list of component objects for dynamic content.
                       Used to fill in template variables ({{1}}, {{2}}, etc.)
        
        Returns:
            The API response as a dict if successful, None on failure.
        
        Example:
            # Simple template with no variables
            await client.send_template("905551234567", "hello_world", "en")
            
            # Template with variables (e.g., "Hello {{1}}, your order {{2}} is ready")
            await client.send_template(
                to="905551234567",
                template="order_ready",
                language="en",
                components=[{
                    "type": "body",
                    "parameters": [
                        {"type": "text", "text": "John"},
                        {"type": "text", "text": "ORD-12345"},
                    ]
                }]
            )
        """
        # Check for missing credentials early
        if not self.config.whatsapp_token or not self.config.phone_number_id:
            print("WARN: Missing WHATSAPP_TOKEN or PHONE_NUMBER_ID, skipping send")
            return None
        
        # Build the template object
        # See: https://developers.facebook.com/docs/whatsapp/cloud-api/messages/template-messages
        template_obj: dict[str, Any] = {
            "name": template,
            "language": {"code": language},
        }
        
        # Add components if provided (for dynamic content)
        if components:
            template_obj["components"] = components
        
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "template",
            "template": template_obj,
        }
        
        try:
            client = self._get_client()
            resp = await client.post(
                self._get_messages_url(),
                headers=self._get_headers(),
                json=payload,
            )
            
            if resp.status_code >= 400:
                print(f"WARN: send_template failed {resp.status_code}: {resp.text}")
                return None
            
            return resp.json()
            
        except httpx.RequestError as e:
            print(f"WARN: send_template network error: {e}")
            return None
    
    async def close(self) -> None:
        """Close the HTTP client and release resources.
        
        Call this when shutting down your application to cleanly close
        the underlying TCP connections.
        
        Example:
            # In FastHTML
            @app.on_event("shutdown")
            async def cleanup():
                await client.close()
        
        Note:
            Safe to call multiple times - subsequent calls do nothing.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None

