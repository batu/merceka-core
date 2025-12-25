"""Data models for WhatsApp messages.

This module defines the data structures used to represent incoming WhatsApp
messages. Supports text messages and image messages.

Usage:
    from merceka_core.wa_bot import Message
    
    # Messages are typically created by parse_webhook_payload() in webhook.py
    msg = Message(
        sender="905551234567",
        text="Hello bot!",
        message_id="wamid.abc123",
        timestamp="1234567890"
    )
    print(f"From {msg.sender}: {msg.text}")
    
    # Image messages have image_id set
    if msg.image_id:
        url = await client.get_media_url(msg.image_id)
"""

from dataclasses import dataclass, field


@dataclass
class Message:
    """A parsed incoming WhatsApp message (text or image).
    
    This represents a message received via the WhatsApp webhook.
    The webhook.py module's parse_webhook_payload() function creates these
    from raw webhook payloads.
    
    Attributes:
        sender: The sender's WhatsApp ID (phone number without + or spaces).
               Example: "905551234567" for a Turkish number.
        text: The message body/content. For text messages, this is the text.
              For image messages, this is the caption (or empty string).
        message_id: Unique identifier assigned by WhatsApp (starts with "wamid.").
                   Useful for deduplication or marking messages as read.
        timestamp: Unix timestamp (as string) when the message was sent.
                  Example: "1702656000" for a message sent Dec 15, 2023.
        image_id: WhatsApp media ID for image messages. None for text messages.
                 Use client.get_media_url(image_id) to get the download URL.
        image_mime_type: MIME type of the image (e.g., "image/jpeg").
                        None for text messages.
    
    Example:
        # Text message handler
        async def handle(client, msg: Message):
            if msg.image_id:
                # Handle image
                url = await client.get_media_url(msg.image_id)
            else:
                # Handle text
                await client.send_text(msg.sender, f"You said: {msg.text}")
    """
    sender: str
    text: str
    message_id: str
    timestamp: str
    image_id: str | None = None
    image_mime_type: str | None = None

