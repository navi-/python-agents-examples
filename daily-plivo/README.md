# Complete Setup Guide: Daily Plivo Phone Bot

This guide provides step-by-step instructions for setting up the Daily Plivo Phone Bot project from scratch.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Account Setup](#account-setup)
- [Project Setup](#project-setup)
- [Configuration](#configuration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Accounts

**Plivo Account**

- Sign up at https://www.plivo.com/
- Verify your account and add payment method
- Purchase a phone number (for both inbound and outbound calls)

**Daily Account**

- Sign up at https://daily.co/
- Get your API key from the dashboard
- Daily provides free tier for development
- Ensure that SIP dial-out is enabled. You may have to reach out to Daily support to enable this.

**AI Service Accounts**

- **OpenAI**: Get API key from https://platform.openai.com/api-keys
- **Deepgram**: Sign up at https://deepgram.com/ and get API key
- **Cartesia**: Sign up at https://cartesia.ai/ and get API key

### Required Software

- Python 3.10 or higher
- `uv` package manager
- ngrok (for local development webhooks)
- Git (for cloning the repository)

## Account Setup

### 1. Plivo Setup

1. Log into your Plivo dashboard
2. Navigate to **Phone Numbers** → **Buy Numbers**
3. Select a number with voice capabilities
4. Note your **Auth ID** and **Auth Token** from the dashboard
5. For inbound calls, you'll configure the answer URL later

### 2. Daily Setup

1. Log into your Daily dashboard
2. Navigate to **Developers** → **API Keys**
3. Create a new API key or use an existing one
4. Copy the API key (you'll need this for `.env`)

### 3. AI Services Setup

**OpenAI:**

1. Create account at https://platform.openai.com/
2. Navigate to API Keys section
3. Create a new secret key
4. Copy the key (starts with `sk-`)

**Deepgram:**

1. Sign up at https://deepgram.com/
2. Navigate to API Keys
3. Create a new API key
4. Copy the key

**Cartesia:**

1. Sign up at https://cartesia.ai/
2. Navigate to API Keys
3. Create a new API key
4. Copy the key

## Project Setup

### 1. Clone or Download the Repository

```bash
git clone <repository-url>
cd daily-plivo
```

Or download and extract the project files.

### 2. Choose Your Project

You have two options:

- **Outbound Calls**: `outbound-call/` - For initiating calls
- **Inbound Calls**: `inbound-call/` - For receiving calls

Navigate to the project directory you want to set up:

```bash
cd outbound-call  # or inbound-call
```

### 3. Install Dependencies

Using `uv`:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 4. Configure Environment Variables

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` and fill in your credentials:

```env
# Plivo Configuration
PLIVO_AUTH_ID=your_plivo_auth_id
PLIVO_AUTH_TOKEN=your_plivo_auth_token
PLIVO_PHONE_NUMBER=+1234567890

# Daily Configuration
DAILY_API_KEY=your_daily_api_key
DAILY_API_URL=https://api.daily.co/v1

# Server Configuration (for webhooks)
SERVER_URL=http://localhost:8000  # For local development
# SERVER_URL=https://your-domain.com  # For production

# AI Services
OPENAI_API_KEY=sk-your_openai_key
DEEPGRAM_API_KEY=your_deepgram_key
CARTESIA_API_KEY=your_cartesia_key
```

**Important**: Never commit the `.env` file to git. It contains sensitive credentials.

## Configuration

### For Outbound Calls

1. Expose your local server using ngrok (required for Plivo webhooks):

```bash
ngrok http 8000
```

2. Copy the HTTPS URL from ngrok (e.g., `https://abc123.ngrok.io`)

3. Update the `SERVER_URL` in your `.env` file:

```env
SERVER_URL=https://abc123.ngrok.io
```

4. Start the server:

```bash
python server.py
```

5. The server will run on `http://localhost:8000` and use the ngrok URL for webhooks

6. For production, you'll need to:
   - Set up a public HTTPS endpoint
   - Update `SERVER_URL` in `.env` to your production URL
   - Configure Plivo webhooks to point to your server

### For Inbound Calls

1. Start the server:

```bash
python server.py
```

2. Expose your local server using ngrok:

```bash
ngrok http 8000
```

3. Copy the HTTPS URL (e.g., `https://abc123.ngrok.io`)

4. Configure your Plivo phone number:
   - Go to Plivo dashboard → **Phone Numbers**
   - Select your phone number
   - Set **Answer URL** to: `https://your-ngrok-url.ngrok.io/plivo-inbound`
   - Set **Answer Method** to: `POST`
   - Optionally set **Hangup URL** to: `https://your-ngrok-url.ngrok.io/plivo-hangup`

## Testing

### Testing Outbound Calls

1. Make sure ngrok is running and `SERVER_URL` in `.env` is updated
2. Make sure the server is running
3. Send a POST request:

```bash
curl -X POST http://localhost:8000/outbound-call \
  -H "Content-Type: application/json" \
  -d '{
    "phone_number": "+1234567890"
  }'
```

4. The call should be initiated and connected to the bot

### Testing Inbound Calls

1. Make sure the server is running
2. Make sure ngrok is running and your Plivo number is configured
3. Call your Plivo phone number from any phone
4. The call should connect to the bot

## Troubleshooting

### Common Issues

**"Module not found" errors**

- Make sure you activated the virtual environment
- Run `uv pip install -r requirements.txt` again

**Plivo authentication errors**

- Verify your Auth ID and Auth Token in `.env`
- Check that credentials are correct in Plivo dashboard

**Daily API errors**

- Verify your Daily API key
- Check that your Daily account is active

**Webhook not receiving calls**

- For inbound: Verify ngrok is running and URL is correct in Plivo
- For outbound: Verify `SERVER_URL` is correct and publicly accessible
- Check server logs for incoming webhook requests

**Bot not responding**

- Check bot logs: `tail -f bot_<room_name>.log`
- Verify all AI service API keys are correct
- Check that bot process started successfully

**Call connects but no audio**

- Verify Daily room was created successfully
- Check that SIP URI is correct
- Verify bot joined the Daily room (check logs)

### Debugging Tips

1. **Check server logs**: Look at the terminal where `server.py` is running
2. **Check bot logs**: Look in `bot_<room_name>.log` files
3. **Check Plivo logs**: Go to Plivo dashboard → **Logs** → **Calls**
4. **Check Daily logs**: Go to Daily dashboard → **Rooms** → Select room → View logs

### Getting Help

- **Daily Support**: https://help.daily.co/
- **Plivo Support**: https://support.plivo.com/
- **OpenAI Support**: https://help.openai.com/
- **Deepgram Support**: https://developers.deepgram.com/support
- **Cartesia Support**: Check their documentation
