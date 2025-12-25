"""WhatsApp bot infrastructure for rapid bot development.

This package provides everything you need to build WhatsApp bots with FastHTML:
- WhatsAppClient: Send text and template messages
- Webhook routes: Handle Meta's verification and incoming messages
- Configuration: MODE-based test/prod environment switching
- Utilities: Safe logging, command normalization

Quick Start:
    from fasthtml.common import FastHTML
    from merceka_core.wa_bot import (
        WhatsAppClient,
        get_config,
        create_webhook_routes,
        Message,
    )
    
    app = FastHTML()
    config = get_config()
    client = WhatsAppClient(config)
    
    async def handle_message(client: WhatsAppClient, msg: Message):
        '''Echo back whatever the user sends.'''
        await client.send_text(msg.sender, f"You said: {msg.text}")
    
    create_webhook_routes(app, client, handle_message, config.verify_token)
    
    # Run with: uvicorn main:app --port 8000

See README.md in this package for detailed setup instructions.
"""

# Configuration
from .config import WhatsAppConfig, get_config, get_mode

# Data models
from .models import Message

# WhatsApp API client
from .client import WhatsAppClient

# Webhook route factory
from .webhook import create_webhook_routes, parse_webhook_payload, MessageHandler

# Utilities
from .utils import redact, normalize_command

# Public API - what gets imported with "from merceka_core.wa_bot import *"
__all__ = [
    # Config
    "WhatsAppConfig",
    "get_config",
    "get_mode",
    # Models
    "Message",
    # Client
    "WhatsAppClient",
    # Webhook
    "create_webhook_routes",
    "parse_webhook_payload",
    "MessageHandler",
    # Utils
    "redact",
    "normalize_command",
]

