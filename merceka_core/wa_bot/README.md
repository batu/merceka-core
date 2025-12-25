# WhatsApp Bot Infrastructure

Build WhatsApp bots quickly with FastHTML.

## What This Package Provides

- **WhatsAppClient**: Send text and template messages via WhatsApp Cloud API
- **Webhook Routes**: Handle Meta's verification and incoming messages
- **Configuration**: MODE-based test/prod environment switching
- **Utilities**: Safe logging, command normalization

## Installation

This package requires `httpx` and `fasthtml` which are not included in merceka_core's dependencies. Install them separately:

```bash
pip install httpx python-fasthtml
# or with uv
uv add httpx python-fasthtml
```

## Quick Start

Create a minimal echo bot in 20 lines:

```python
# main.py
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
    """Echo back whatever the user sends."""
    await client.send_text(msg.sender, f"You said: {msg.text}")

create_webhook_routes(app, client, handle_message, config.verify_token)

# Run with: uvicorn main:app --port 8000
```

## Meta Developer Console Setup

### Step 1: Create a Meta App

1. Go to [Meta Developer Console](https://developers.facebook.com/)
2. Click "My Apps" → "Create App"
3. Select "Business" type
4. Fill in app name and contact email
5. On the dashboard, find "WhatsApp" and click "Set Up"

### Step 2: Get Your Credentials

Go to **WhatsApp** → **API Setup** in the left sidebar.

You'll find:

| Value | Where to Find It | Env Variable |
|-------|------------------|--------------|
| **Phone Number ID** | Under "From" dropdown, the number in parentheses | `TEST_PHONE_NUMBER_ID` |
| **WhatsApp Token** | "Temporary access token" section, click "Generate" | `WHATSAPP_TOKEN` |
| **WABA ID** | In webhook payloads, or Business Settings | `TEST_WABA_ID` |

**Note**: Temporary tokens expire in ~24 hours. For production, create a System User token.

### Step 3: Create Your .env File

```bash
# .env

# ============================================
# ENVIRONMENT MODE: "test" or "prod"
# ============================================
MODE=test

# ============================================
# SHARED CONFIG (same for both environments)
# ============================================
VERIFY_TOKEN=your_secret_verify_token
WHATSAPP_TOKEN=EAAG...your_token_here
GRAPH_VERSION=v24.0

# ============================================
# TEST ENVIRONMENT
# ============================================
TEST_PHONE_NUMBER_ID=123456789012345
TEST_WABA_ID=123456789012345

# ============================================
# PRODUCTION ENVIRONMENT (fill in later)
# ============================================
PROD_PHONE_NUMBER_ID=
PROD_WABA_ID=
```

### Step 4: Set Up Public URL with Tailscale Funnel

Meta needs to reach your webhook from the internet. Tailscale Funnel creates a public URL for your local server.

1. Install Tailscale: https://tailscale.com/download

2. Enable Funnel (one-time):
   ```bash
   tailscale up
   tailscale funnel 8000
   ```

3. Note your public URL (e.g., `https://your-machine.your-tailnet.ts.net`)

4. Verify it works:
   ```bash
   # Start your server first
   uvicorn main:app --port 8000
   
   # In another terminal, test the public URL
   curl https://your-machine.your-tailnet.ts.net/webhook?hub.mode=subscribe&hub.verify_token=your_secret_verify_token&hub.challenge=TEST
   # Should return: TEST
   ```

### Step 5: Configure Webhook in Meta Console

1. Go to **WhatsApp** → **Configuration** in Meta Developer Console
2. Under "Webhook", click **Edit**
3. Enter:
   - **Callback URL**: `https://your-machine.your-tailnet.ts.net/webhook`
   - **Verify Token**: Same value as `VERIFY_TOKEN` in your .env
4. Click **Verify and Save**
5. Subscribe to **messages** field

### Step 6: Subscribe App to WABA (Critical!)

This step is often missed. Without it, test payloads work but real messages don't arrive.

```bash
source .env
curl -X POST "https://graph.facebook.com/$GRAPH_VERSION/$TEST_WABA_ID/subscribed_apps" \
  -H "Authorization: Bearer $WHATSAPP_TOKEN"
```

Expected response:
```json
{"success": true}
```

### Step 7: Add Test Phone Numbers

With a test account, you can only message phone numbers you've added.

1. Go to **WhatsApp** → **API Setup**
2. Under "To", click **Manage phone number list**
3. Add phone numbers you want to message

## Environment Switching

Switch between test and production by changing `MODE`:

```bash
# .env
MODE=test   # Uses TEST_PHONE_NUMBER_ID, TEST_WABA_ID
MODE=prod   # Uses PROD_PHONE_NUMBER_ID, PROD_WABA_ID
```

Then restart your server.

## Template Messages

Template messages can be sent outside the 24-hour messaging window. They must be pre-approved in Meta Business Manager.

```python
# Simple template (no variables)
await client.send_template(
    to="905551234567",
    template="hello_world",
    language="en"
)

# Template with variables
# If your template is: "Hello {{1}}, your order {{2}} is ready"
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
```

## Testing

### Test 1: Server Health

```bash
curl http://127.0.0.1:8000/health
# Should return: ok (if you have a /health route)
```

### Test 2: Webhook Verification

```bash
source .env
curl "http://127.0.0.1:8000/webhook?hub.mode=subscribe&hub.verify_token=$VERIFY_TOKEN&hub.challenge=TEST"
# Should return: TEST
```

### Test 3: Simulate Incoming Message

```bash
curl -X POST http://127.0.0.1:8000/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "object": "whatsapp_business_account",
    "entry": [{
      "id": "test",
      "changes": [{
        "field": "messages",
        "value": {
          "messages": [{
            "from": "905551234567",
            "id": "wamid.test123",
            "timestamp": "1702656000",
            "type": "text",
            "text": {"body": "Hello bot!"}
          }]
        }
      }]
    }]
  }'
# Should return: ok
# Check your terminal for handler output
```

### Test 4: Check WABA Subscription

```bash
source .env
curl "https://graph.facebook.com/$GRAPH_VERSION/$TEST_WABA_ID/subscribed_apps" \
  -H "Authorization: Bearer $WHATSAPP_TOKEN"
```

## Common Errors

### "Webhook verified but real messages don't arrive"

**Cause**: App not subscribed to WABA.

**Fix**: Run the subscription curl command from Step 6.

### "Recipient phone number not in allowed list"

**Cause**: Using test phone number, can only message pre-approved numbers.

**Fix**: Add recipient to allowed list in Meta Console, or upgrade to production.

### "Invalid OAuth access token"

**Cause**: Token expired (temporary tokens last ~24 hours).

**Fix**: Generate a new token in Meta Developer Console.

### "Missing WHATSAPP_TOKEN or PHONE_NUMBER_ID"

**Cause**: Environment variables not loaded.

**Fix**: Make sure .env file exists and contains the values. Check that `python-dotenv` is installed.

## API Reference

### WhatsAppConfig

```python
@dataclass
class WhatsAppConfig:
    phone_number_id: str   # From Meta Console
    whatsapp_token: str    # Bearer token for API
    verify_token: str      # Your webhook verification secret
    waba_id: str           # WhatsApp Business Account ID
    graph_version: str     # Default: "v24.0"
```

### get_config() → WhatsAppConfig

Load configuration from environment variables, respecting MODE prefix.

### Message

```python
@dataclass
class Message:
    sender: str      # WhatsApp ID (e.g., "905551234567")
    text: str        # Message body
    message_id: str  # Unique ID (e.g., "wamid.abc123")
    timestamp: str   # Unix timestamp as string
```

### WhatsAppClient

```python
client = WhatsAppClient(config)

# Send text message
await client.send_text(to="905551234567", body="Hello!")

# Send template message
await client.send_template(to="905551234567", template="hello_world", language="en")

# Clean up (call on shutdown)
await client.close()
```

### create_webhook_routes()

```python
def create_webhook_routes(
    app: FastHTML,           # Your FastHTML app
    client: WhatsAppClient,  # Client for sending replies
    handler: MessageHandler, # async def handler(client, msg)
    verify_token: str,       # Must match Meta Console
    path: str = "/webhook"   # URL path
) -> None
```

### Utilities

```python
# Safe logging (hides tokens)
from merceka_core.wa_bot import redact
print(redact({"token": "secret", "user": "john"}))
# {"token": "***REDACTED***", "user": "john"}

# Command normalization
from merceka_core.wa_bot import normalize_command
if normalize_command(msg.text) == "help":
    await send_help()
```

## Full Example: Command Bot

```python
# main.py - A bot with multiple commands
from fasthtml.common import FastHTML, Response
from merceka_core.wa_bot import (
    WhatsAppClient,
    get_config,
    create_webhook_routes,
    Message,
    normalize_command,
)

app = FastHTML()
config = get_config()
client = WhatsAppClient(config)

# Simple in-memory state (use a database for production)
user_names: dict[str, str] = {}

async def handle_message(client: WhatsAppClient, msg: Message):
    """Handle incoming messages with commands."""
    command = normalize_command(msg.text)
    
    if command == "help":
        await client.send_text(msg.sender, 
            "Available commands:\n"
            "• help - Show this message\n"
            "• hello - Get a greeting\n"
            "• name <your name> - Set your name\n"
            "• whoami - Show your saved name"
        )
    
    elif command == "hello":
        name = user_names.get(msg.sender, "stranger")
        await client.send_text(msg.sender, f"Hello, {name}! 👋")
    
    elif command.startswith("name "):
        name = msg.text[5:].strip()  # Get everything after "name "
        user_names[msg.sender] = name
        await client.send_text(msg.sender, f"Nice to meet you, {name}!")
    
    elif command == "whoami":
        name = user_names.get(msg.sender)
        if name:
            await client.send_text(msg.sender, f"You are {name}")
        else:
            await client.send_text(msg.sender, "I don't know your name yet. Use: name <your name>")
    
    else:
        await client.send_text(msg.sender, f"Unknown command. Type 'help' for options.")

# Health check endpoint
@app.get("/health")
def health():
    return Response("ok", media_type="text/plain")

# Register webhook routes
create_webhook_routes(app, client, handle_message, config.verify_token)

# Clean up on shutdown
@app.on_event("shutdown")
async def shutdown():
    await client.close()

# Run with: uvicorn main:app --port 8000
```

