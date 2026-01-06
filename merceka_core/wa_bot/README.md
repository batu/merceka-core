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

---

# Meta Developer Console Setup

> **Last Updated**: January 2026
> 
> The Meta Developer Console UI changes frequently. This guide documents the current navigation paths.

## Overview: Two Setup Paths

| Path | Use Case | Phone Number |
|------|----------|--------------|
| **Test Mode** | Development & testing | Meta provides a temporary test number |
| **Production Mode** | Real users | Your own verified phone number |

Both paths require the same steps, but credentials differ.

---

## Step 1: Create a Meta App

1. Go to [Meta Developer Console](https://developers.facebook.com/)
2. Click **"My Apps"** → **"Create App"**
3. Select use case: **"Other"** → App type: **"Business"**
4. Fill in:
   - App name (e.g., "My WhatsApp Bot")
   - Contact email
   - Business portfolio (select or create one)
5. Click **"Create App"**

After creation, you'll land on the app dashboard.

### Add WhatsApp Product

1. On the dashboard, find **"Add products to your app"**
2. Find **"WhatsApp"** and click **"Set up"**
3. This creates a test WhatsApp Business Account automatically

---

## Step 2: Navigate the WhatsApp Section

After adding WhatsApp, you'll see a new navigation structure:

```
Use cases → Customize
└── Connect on WhatsApp (dropdown)
    ├── Permissions and features
    ├── Quickstart          ← Overview, links to phone management
    ├── API Testing         ← Test credentials, send test messages
    ├── Configuration       ← Webhook setup
    ├── Resources
    └── Tech Provider onboarding
```

### Key Pages

| Page | What's There |
|------|--------------|
| **Quickstart** | Links to WhatsApp Manager, phone numbers, permanent token setup |
| **API Testing** | Test WABA ID, Phone Number ID, temporary token, send test messages |
| **Configuration** | Webhook URL, verify token, subscribe to message fields |

---

## Step 3: Get Test Credentials (Quick Start)

For initial development, use the test credentials:

1. Go to **API Testing** tab
2. Find **"2. Select a 'From' phone number"**
3. You'll see:
   - **WhatsApp Business Account ID**: `1234567890123456` (TEST_WABA_ID)
   - **Phone number ID**: `9876543210987654` (TEST_PHONE_NUMBER_ID)

4. Under **"1. Generate a temporary access token"**, click **"Generate access token"**
   - This token expires in ~60 minutes
   - Use for testing only

---

## Step 4: Create a Permanent Access Token (Production)

Temporary tokens expire. For production, create a System User token:

### 4a. Access Business Settings

1. In **Quickstart**, find **"Create a permanent access token"**
2. Click **"Business Settings"** button
3. Or go directly to: [business.facebook.com/settings](https://business.facebook.com/settings)

### 4b. Create a System User

1. Navigate to **Users** → **System users** (left sidebar)
2. Click **"+ Add"** button
3. Enter:
   - **Name**: e.g., "WhatsApp Bot"
   - **Role**: **Admin**
4. Click **"Create system user"**

### 4c. Assign Assets to System User

This is critical! The system user needs access to both the App AND the WhatsApp Account.

1. Click on your system user
2. Click **"⋯"** (three dots) → **"Assign assets"**

**Assign App:**
1. Select asset type: **Apps**
2. Check your app (e.g., "My WhatsApp Bot")
3. Enable: **Full control** → **Manage app**
4. Click **"Assign assets"**

**Assign WhatsApp Account:**
1. Click **"Assign assets"** again
2. Select asset type: **WhatsApp accounts**
3. Check your production WABA (e.g., "Your Business Name")
4. Enable: **Full control** → **Everything**
5. Click **"Assign assets"**

> ⚠️ **Gotcha**: You may need to **refresh the page** after assigning assets before the token generation shows correct permissions.

### 4d. Generate the Token

1. Click **"Generate token"** button on the system user
2. Select your app
3. Set expiration: **Never** (for production)
4. Select permissions:
   - ✅ `whatsapp_business_messaging` (Required - send/receive messages)
   - ✅ `whatsapp_business_management` (Required - manage account)
   - ⬜ `whatsapp_business_manage_events` (Optional - webhook management)
   - ⬜ `manage_app_solution` (Not needed - for BSPs only)
5. Click **"Generate token"**
6. **Copy and save the token securely** - you won't see it again!

---

## Step 5: Add Your Production Phone Number

For production, you need your own phone number (not the test number).

### Prerequisites

- Phone number must **NOT** be registered with any WhatsApp account (personal or business app)
- If it is, delete that WhatsApp account first
- Number must be able to receive SMS or voice calls for verification

### Add the Number

1. Go to **Quickstart** tab
2. Scroll down to **"WhatsApp Business"** section
3. Click **"Manage phone numbers"**
4. In WhatsApp Manager, select your production WABA from the dropdown (not "Test WhatsApp Business Account")
5. Click **"Add phone number"** (top right)
6. Fill in:
   - **Display name**: What users see (e.g., "My Business")
   - **Category**: Select appropriate business category
   - **Phone number**: Your number in international format
7. Verify via **SMS** or **Voice call**

After verification, note your:
- **Phone Number ID**: Shown in the phone number details
- **WABA ID**: Shown in the account selector dropdown

---

## Step 6: Create Your .env File

```bash
# .env

# ============================================
# ENVIRONMENT MODE: "test" or "prod"
# ============================================
MODE=test

# ============================================
# SHARED CONFIG (same for both environments)
# ============================================
VERIFY_TOKEN=your_secret_verify_token_here
WHATSAPP_TOKEN=EAAxxxx...your_permanent_token
GRAPH_VERSION=v24.0

# ============================================
# TEST ENVIRONMENT (from API Testing page)
# ============================================
TEST_PHONE_NUMBER_ID=980482958475149
TEST_WABA_ID=1207742824654086

# ============================================
# PRODUCTION ENVIRONMENT (from WhatsApp Manager)
# ============================================
PROD_PHONE_NUMBER_ID=935995412929932
PROD_WABA_ID=807310585618264
```

---

## Step 7: Set Up Public URL with Tailscale Funnel

Meta's webhooks need to reach your server from the internet. Tailscale Funnel creates a secure public URL.

### Install Tailscale

```bash
# Linux
curl -fsSL https://tailscale.com/install.sh | sh

# macOS
brew install tailscale

# Or download from: https://tailscale.com/download
```

### Enable Funnel

```bash
# Connect to Tailscale
tailscale up

# Enable funnel on port 8000
tailscale funnel 8000
```

### Get Your Public URL

```bash
tailscale funnel status
```

Output shows your URL, e.g.: `https://your-machine.tail12345.ts.net`

Your webhook URL will be: `https://your-machine.tail12345.ts.net/webhook`

### Verify It Works

```bash
# Start your server first
uvicorn main:app --port 8000

# In another terminal, test through the public URL
curl "https://your-machine.tail12345.ts.net/webhook?hub.mode=subscribe&hub.verify_token=your_secret_verify_token_here&hub.challenge=TEST"
# Should return: TEST
```

---

## Step 8: Configure Webhook in Meta Console

1. Go to your app → **WhatsApp** → **Configuration** tab
2. Scroll to **"Subscribe to webhooks"**
3. Enter:
   - **Callback URL**: `https://your-machine.tail12345.ts.net/webhook`
   - **Verify token**: Same as `VERIFY_TOKEN` in your .env
4. Leave **"Attach a client certificate"** OFF
5. Click **"Verify and save"**

### Subscribe to Webhook Fields

After verification, you'll see a list of webhook fields:

| Field | Subscribe? | Purpose |
|-------|-----------|---------|
| `messages` | ✅ **Yes** | Receive incoming messages |
| `message_template_status_update` | Optional | Template approval notifications |
| `messaging_handovers` | Optional | Multi-bot handover |

Toggle **"messages"** to **Subscribed**.

---

## Step 9: Subscribe App to WABA (Critical!)

> ⚠️ **This step is often missed!** Without it, webhook verification works but real messages never arrive.

```bash
source .env

# For test environment
curl -X POST "https://graph.facebook.com/$GRAPH_VERSION/$TEST_WABA_ID/subscribed_apps" \
  -H "Authorization: Bearer $WHATSAPP_TOKEN"

# For production environment
curl -X POST "https://graph.facebook.com/$GRAPH_VERSION/$PROD_WABA_ID/subscribed_apps" \
  -H "Authorization: Bearer $WHATSAPP_TOKEN"
```

Expected response:
```json
{"success": true}
```

### Verify Subscription

```bash
curl "https://graph.facebook.com/$GRAPH_VERSION/$PROD_WABA_ID/subscribed_apps" \
  -H "Authorization: Bearer $WHATSAPP_TOKEN"
```

---

## Step 10: Test Your Bot

Send a WhatsApp message to your bot's phone number. You should receive an echo response!

---

# Environment Switching

Switch between test and production by changing `MODE`:

```bash
# .env
MODE=test   # Uses TEST_PHONE_NUMBER_ID, TEST_WABA_ID
MODE=prod   # Uses PROD_PHONE_NUMBER_ID, PROD_WABA_ID
```

Then restart your server.

---

# Credentials Reference

## Where to Find Each Credential

| Credential | Location | Notes |
|------------|----------|-------|
| **Test Phone Number ID** | API Testing → "2. Select a 'From' phone number" | Under the dropdown |
| **Test WABA ID** | API Testing → "2. Select a 'From' phone number" | "WhatsApp Business Account ID" |
| **Prod Phone Number ID** | WhatsApp Manager → Phone numbers → Click number | In the details panel |
| **Prod WABA ID** | WhatsApp Manager → Account dropdown | The ID shown under account name |
| **Temporary Token** | API Testing → "1. Generate a temporary access token" | Expires in ~60 min |
| **Permanent Token** | Business Settings → System Users → Generate token | Never expires |
| **Verify Token** | You create this | Any secret string you choose |

---

# Common Errors

### "Webhook verified but real messages don't arrive"

**Cause**: App not subscribed to WABA.

**Fix**: Run the subscription curl command from Step 9.

### "No permissions available" when generating token

**Cause**: System user doesn't have assets assigned.

**Fix**: Assign both the App AND the WhatsApp Account to the system user (Step 4c). Refresh the page after assigning.

### "Recipient phone number not in allowed list"

**Cause**: Test mode only allows pre-approved recipient numbers.

**Fix**: In API Testing, under "3. Add a recipient phone number", add the recipient. Or switch to production mode.

### "Invalid OAuth access token"

**Cause**: Token expired (temporary tokens last ~60 minutes).

**Fix**: Generate a new token, or use a permanent System User token.

### "Message failed to send" / Error 131030

**Cause**: 24-hour messaging window expired.

**Fix**: User must message you first, or use an approved template message.

---

# Template Messages

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

---

# Testing Locally

### Test 1: Server Health

```bash
curl http://127.0.0.1:8000/health
# Should return: ok
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

---

# API Reference

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

---

# Full Example: Command Bot

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

---

# Important Notes

### One Phone Number = One Bot

Each WhatsApp phone number can only be connected to one webhook. If you need multiple bots:
- Use separate phone numbers, OR
- Build multiple features into one bot with command routing

### Phone Number Requirements

- Cannot be registered with WhatsApp (personal) or WhatsApp Business App
- Must be able to receive SMS or voice calls
- One number per WhatsApp Business Account webhook

### Rate Limits & Messaging Tiers

- New accounts start with limited messaging (1,000 business-initiated conversations/day)
- Tier increases with quality rating and volume
- User-initiated messages (replies within 24h) have higher limits

### Business Verification

- Required for: higher messaging limits, official business account badge
- Complete in Meta Business Suite → Business Settings → Business Info
