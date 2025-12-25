"""Data models for WhatsApp messages.

This module defines the data structures used to represent incoming WhatsApp
messages. Currently focused on text messages for simplicity.

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
"""

from dataclasses import dataclass


@dataclass
class Message:
    """A parsed incoming WhatsApp text message.
    
    This represents a single text message received via the WhatsApp webhook.
    The webhook.py module's parse_webhook_payload() function creates these
    from raw webhook payloads.
    
    Attributes:
        sender: The sender's WhatsApp ID (phone number without + or spaces).
               Example: "905551234567" for a Turkish number.
        text: The message body/content. This is the actual text the user sent.
        message_id: Unique identifier assigned by WhatsApp (starts with "wamid.").
                   Useful for deduplication or marking messages as read.
        timestamp: Unix timestamp (as string) when the message was sent.
                  Example: "1702656000" for a message sent Dec 15, 2023.
    
    Example:
        # Echo bot handler
        async def handle(client, msg: Message):
            await client.send_text(msg.sender, f"You said: {msg.text}")
    """
    sender: str
    text: str
    message_id: str
    timestamp: str

