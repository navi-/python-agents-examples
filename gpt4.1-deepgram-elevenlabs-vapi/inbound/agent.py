"""Inbound voice agent — Vapi assistant configuration and webhook handling.

Vapi is a hosted orchestrator that manages the entire voice pipeline:
- Deepgram Nova-3 for speech-to-text
- OpenAI GPT-4.1 for conversation intelligence
- ElevenLabs for text-to-speech
- Server-side VAD and turn detection

Plivo connects to Vapi via SIP trunking (not WebSocket streaming).
Our server only handles Vapi webhook events (tool calls, status updates, etc.).
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Agent configuration
VAPI_PRIVATE_KEY = os.getenv("VAPI_PRIVATE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4.1")
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text().strip()
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", SYSTEM_PROMPT)

# =============================================================================
# Tool Definitions (for Vapi assistant config)
# =============================================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Look up the status of a customer's order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {
                        "type": "string",
                        "description": "Order number (usually starts with TF-)",
                    },
                    "email": {
                        "type": "string",
                        "description": "Customer's email if order number unavailable",
                    },
                },
            },
        },
        "server": {"url": None},  # Uses assistant's server.url
    },
    {
        "type": "function",
        "function": {
            "name": "send_sms",
            "description": "Send a text message to the customer's phone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {"type": "string", "description": "Phone number"},
                    "message": {"type": "string", "description": "Message content"},
                },
                "required": ["phone_number", "message"],
            },
        },
        "server": {"url": None},
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_callback",
            "description": "Schedule a callback from a specialist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {"type": "string", "description": "Phone number"},
                    "reason": {"type": "string", "description": "Why callback is needed"},
                    "preferred_time": {"type": "string", "description": "Preferred time"},
                    "department": {"type": "string", "description": "Department"},
                },
                "required": ["phone_number", "reason", "department"],
            },
        },
        "server": {"url": None},
    },
    # Native Vapi tool types — no custom handler needed
    {
        "type": "endCall",
    },
    {
        "type": "transferCall",
        "destinations": [
            {
                "type": "number",
                "number": "+13305263709",
                "message": "I'm transferring you to a specialist now. Please hold.",
                "description": "Transfer to customer support specialist",
            },
        ],
    },
]


# =============================================================================
# Tool Execution Functions
# =============================================================================


async def check_order_status(
    order_number: str | None, email: str | None
) -> dict[str, Any]:
    """Look up order status. Replace with your actual implementation."""
    logger.info(f"Checking order: number={order_number}, email={email}")

    if not order_number and not email:
        return {"status": "error", "message": "Need order number or email"}

    statuses = [
        {
            "status": "shipped",
            "order_number": order_number or f"TF-{random.randint(100000, 999999)}",
            "shipping_carrier": "FedEx",
            "tracking_number": f"FX{random.randint(1000000000, 9999999999)}",
            "estimated_delivery": (datetime.now() + timedelta(days=2)).strftime("%B %d"),
            "items": "TechFlow Pro Annual Subscription",
        },
        {
            "status": "processing",
            "order_number": order_number or f"TF-{random.randint(100000, 999999)}",
            "message": "Order is being prepared and will ship within 24 hours",
            "items": "TechFlow Teams License (5 seats)",
        },
        {
            "status": "delivered",
            "order_number": order_number or f"TF-{random.randint(100000, 999999)}",
            "delivered_date": (datetime.now() - timedelta(days=1)).strftime("%B %d"),
            "signed_by": "Front Desk",
            "items": "TechFlow Enterprise Setup Kit",
        },
    ]
    return random.choice(statuses)


async def send_sms(phone_number: str, message: str) -> dict[str, Any]:
    """Send SMS to customer. Replace with your actual implementation."""
    logger.info(f"Sending SMS to {phone_number}: {message[:50]}...")

    if not phone_number:
        return {"status": "error", "message": "Phone number required"}

    return {
        "status": "sent",
        "phone_number": phone_number,
        "message_preview": message[:50] + "..." if len(message) > 50 else message,
        "confirmation_id": f"SMS{random.randint(100000, 999999)}",
    }


async def schedule_callback(
    phone_number: str, reason: str, preferred_time: str, department: str
) -> dict[str, Any]:
    """Schedule a callback. Replace with your actual implementation."""
    logger.info(f"Scheduling callback: {phone_number}, {department}")

    if not phone_number:
        return {"status": "error", "message": "Phone number required"}

    return {
        "status": "scheduled",
        "callback_id": f"CB{random.randint(100000, 999999)}",
        "phone_number": phone_number,
        "department": department,
        "scheduled_time": preferred_time or "within 2 business hours",
        "reason": reason,
    }


# =============================================================================
# Webhook Event Handlers
# =============================================================================

TOOL_HANDLERS = {
    "check_order_status": check_order_status,
    "send_sms": send_sms,
    "schedule_callback": schedule_callback,
}


async def handle_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute tool calls from Vapi and return results.

    Vapi sends tool calls with top-level ``name`` and ``arguments`` fields
    (not nested under a ``function`` key).
    """
    tool_call_list = message.get("toolCallList", [])
    results = []

    for tool_call in tool_call_list:
        name = tool_call.get("name", "")
        tool_call_id = tool_call.get("id", "")

        try:
            arguments = tool_call.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            arguments = {}

        logger.info(f"Tool call: {name} with args: {arguments}")

        handler = TOOL_HANDLERS.get(name)
        if handler:
            result = await handler(**arguments)
        else:
            result = {"error": f"Unknown function: {name}"}

        results.append({
            "name": name,
            "toolCallId": tool_call_id,
            "result": json.dumps(result),
        })

    return results


def build_assistant_config(server_url: str) -> dict[str, Any]:
    """Build a transient Vapi assistant configuration for inbound calls.

    This is returned in response to the assistant-request webhook so Vapi
    knows how to configure the call pipeline dynamically.

    Caller context (phone number, time) is provided via Vapi template variables
    in the system prompt — no server-side interpolation needed.
    """
    config: dict[str, Any] = {
        "firstMessage": (
            "Hello! Thank you for calling TechFlow. This is Alex. "
            "How can I help you today?"
        ),
        "firstMessageMode": "assistant-speaks-first",
        "transcriber": {
            "provider": "deepgram",
            "model": DEEPGRAM_MODEL,
            "language": "en",
        },
        "model": {
            "provider": "openai",
            "model": GPT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            ],
            "temperature": 0.7,
            "tools": TOOL_DEFINITIONS,
        },
        "voice": {
            "provider": "11labs",
            "voiceId": ELEVENLABS_VOICE_ID,
            "model": ELEVENLABS_MODEL,
            "stability": 0.5,
            "similarityBoost": 0.75,
        },
        "server": {"url": server_url},
        "serverMessages": [
            "tool-calls",
            "status-update",
            "end-of-call-report",
            "transcript",
            "user-interrupted",
        ],
        "endCallMessage": "Thank you for calling TechFlow. Have a great day!",
        "endCallPhrases": ["goodbye", "bye bye", "that's all", "nothing else"],
        "recordingEnabled": True,
        "backgroundDenoisingEnabled": True,
        "analysisPlan": {
            "summaryPlan": {"enabled": True},
            "successEvaluationPlan": {
                "enabled": True,
                "rubric": "AutomaticRubric",
            },
        },
        # VAD and turn detection configuration
        "stopSpeakingPlan": {
            "numWords": 2,
            "voiceSeconds": 0.2,
            "backoffSeconds": 1.0,
        },
        "startSpeakingPlan": {
            "waitSeconds": 0.4,
            "transcriptionEndpointingPlan": {
                "onPunctuationSeconds": 0.1,
                "onNoPunctuationSeconds": 1.5,
                "onNumberSeconds": 0.5,
            },
        },
    }

    return config
