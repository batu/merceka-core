"""Webhook route factory for FastHTML apps.

This module provides functions to set up WhatsApp webhook routes in a FastHTML
application. It handles both the verification handshake (GET) and incoming
message processing (POST).

How WhatsApp Webhooks Work:
1. You register a webhook URL in Meta Developer Console
2. Meta sends a GET request with a challenge to verify you own the URL
3. Once verified, Meta sends POST requests with incoming messages

Usage:
    from fasthtml.common import FastHTML
    from merceka_core.wa_bot import (
        WhatsAppClient, get_config, Message, create_webhook_routes
    )
    
    app = FastHTML()
    config = get_config()
    client = WhatsAppClient(config)
    
    async def handle_message(client: WhatsAppClient, msg: Message):
        await client.send_text(msg.sender, f"You said: {msg.text}")
    
    create_webhook_routes(app, client, handle_message, config.verify_token)

Dependencies:
    - fasthtml: pip install python-fasthtml
"""

import json
from collections.abc import Awaitable, Callable

from fasthtml.common import FastHTML, Request, Response

from .client import WhatsAppClient
from .models import Message

# Type alias for message handler functions
# Handler receives the client (to send replies) and the parsed message
MessageHandler = Callable[[WhatsAppClient, Message], Awaitable[None]]


def parse_webhook_payload(payload: dict) -> list[Message]:
    """Extract Message objects from a WhatsApp webhook payload.
    
    WhatsApp webhook payloads have a deeply nested structure. This function
    navigates that structure and extracts text and image messages.
    
    Payload structure for text messages:
    {
        "entry": [{"changes": [{"value": {"messages": [{
            "from": "905551234567",
            "id": "wamid.abc123",
            "timestamp": "1702656000",
            "type": "text",
            "text": {"body": "Hello!"}
        }]}}]}]
    }
    
    Payload structure for image messages:
    {
        "entry": [{"changes": [{"value": {"messages": [{
            "from": "905551234567",
            "id": "wamid.abc123",
            "timestamp": "1702656000",
            "type": "image",
            "image": {"id": "media123", "mime_type": "image/jpeg", "caption": "optional"}
        }]}}]}]
    }
    
    Args:
        payload: The raw webhook payload as a dict (already parsed from JSON)
    
    Returns:
        List of Message objects for text and image messages. May be empty if:
        - Payload has unexpected structure
        - No messages in the payload (e.g., only status updates)
        - Messages are unsupported types (audio, document, etc.)
    
    Example:
        messages = parse_webhook_payload(payload)
        for msg in messages:
            if msg.image_id:
                print(f"Image from {msg.sender}")
            else:
                print(f"Text from {msg.sender}: {msg.text}")
    """
    messages: list[Message] = []
    
    entries = payload.get("entry", [])
    
    for entry in entries:
        if not isinstance(entry, dict):
            continue
            
        changes = entry.get("changes", [])
        
        for change in changes:
            if not isinstance(change, dict):
                continue
                
            value = change.get("value", {})
            
            if not isinstance(value, dict):
                continue
                
            raw_messages = value.get("messages", [])
            
            for msg in raw_messages:
                if not isinstance(msg, dict):
                    continue
                
                msg_type = msg.get("type")
                sender = msg.get("from")
                
                if not sender:
                    continue
                
                # Handle text messages
                if msg_type == "text":
                    text_obj = msg.get("text", {})
                    text = text_obj.get("body", "") if isinstance(text_obj, dict) else ""
                    
                    messages.append(Message(
                        sender=sender,
                        text=text,
                        message_id=msg.get("id", ""),
                        timestamp=msg.get("timestamp", ""),
                    ))
                
                # Handle image messages
                elif msg_type == "image":
                    image_obj = msg.get("image", {})
                    if not isinstance(image_obj, dict):
                        continue
                    
                    image_id = image_obj.get("id")
                    if not image_id:
                        continue
                    
                    # Caption is optional text that comes with an image
                    caption = image_obj.get("caption", "")
                    
                    messages.append(Message(
                        sender=sender,
                        text=caption,
                        message_id=msg.get("id", ""),
                        timestamp=msg.get("timestamp", ""),
                        image_id=image_id,
                        image_mime_type=image_obj.get("mime_type"),
                    ))
                
                # Other types (audio, document, sticker, etc.) are silently skipped
    
    return messages


def create_webhook_routes(
    app: FastHTML,
    client: WhatsAppClient,
    handler: MessageHandler,
    verify_token: str,
    path: str = "/webhook",
) -> None:
    """Register webhook routes on a FastHTML app.
    
    This adds two routes to your app:
    1. GET {path} - Webhook verification (Meta's challenge-response)
    2. POST {path} - Receive incoming messages
    
    Args:
        app: Your FastHTML application instance
        client: WhatsAppClient for sending replies
        handler: Async function called for each incoming text message.
                Signature: async def handler(client, message) -> None
        verify_token: The token you set in Meta Developer Console.
                     Must match exactly or verification will fail.
        path: URL path for the webhook (default: "/webhook")
    
    Example:
        app = FastHTML()
        config = get_config()
        client = WhatsAppClient(config)
        
        async def echo_handler(client: WhatsAppClient, msg: Message):
            await client.send_text(msg.sender, f"Echo: {msg.text}")
        
        create_webhook_routes(app, client, echo_handler, config.verify_token)
        
        # Now your app has:
        # GET /webhook - for Meta verification
        # POST /webhook - for incoming messages
    
    Note:
        The handler is called for each text message. Non-text messages
        (images, audio, etc.) are silently skipped.
    
    TODO: Add signature verification (X-Hub-Signature-256) for production security
    """
    
    @app.get(path)
    def webhook_verify(req: Request) -> Response:
        """Handle Meta's webhook verification request.
        
        When you configure a webhook URL in Meta Developer Console, Meta sends
        a GET request with these query parameters:
        - hub.mode: Always "subscribe"
        - hub.verify_token: The token you entered in Meta console
        - hub.challenge: A random string Meta wants you to echo back
        
        If the token matches, we return the challenge to prove we own this URL.
        """
        # Extract query parameters
        mode = req.query_params.get("hub.mode")
        token = req.query_params.get("hub.verify_token")
        challenge = req.query_params.get("hub.challenge")
        
        # Verify the request
        # All three conditions must be true for successful verification
        if mode == "subscribe" and token == verify_token and challenge is not None:
            # Success! Echo back the challenge
            print(f"WEBHOOK: Verification successful")
            return Response(challenge, media_type="text/plain")
        
        # Verification failed - wrong token or missing parameters
        print(f"WEBHOOK: Verification failed (mode={mode}, token_match={token == verify_token})")
        return Response("forbidden", status_code=403, media_type="text/plain")
    
    @app.post(path)
    async def webhook_receive(req: Request) -> Response:
        """Handle incoming webhook events from WhatsApp.
        
        Meta sends POST requests here whenever something happens:
        - New message received
        - Message status update (sent, delivered, read)
        - Errors
        
        We parse the payload, extract text messages, and call the handler
        for each one. Non-text messages are silently skipped.
        
        Important: We always return 200 OK quickly. Meta will retry if we
        don't respond within ~15 seconds, which can cause duplicate messages.
        """
        print(f"WEBHOOK: POST received from {req.client}")
        
        # Parse the JSON payload
        try:
            body = await req.body()
            payload = json.loads(body)
        except Exception as e:
            print(f"WEBHOOK: Invalid JSON payload: {e}")
            # Still return 200 to prevent Meta retries
            return Response("ok", status_code=200, media_type="text/plain")
        
        # Extract messages from the payload
        messages = parse_webhook_payload(payload)
        
        # Process each message
        for msg in messages:
            try:
                # Call the user's handler
                await handler(client, msg)
            except Exception as e:
                # Log but don't crash - we want to process other messages
                print(f"WEBHOOK: Handler error for message {msg.message_id}: {e}")
        
        # Always return 200 OK to acknowledge receipt
        # This prevents Meta from retrying the webhook
        return Response("ok", status_code=200, media_type="text/plain")

