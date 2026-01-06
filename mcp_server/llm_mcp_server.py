"""
LLM MCP Server - Exposes LLM cascade via Model Context Protocol.

This server wraps the LLM cascade (DeepSeek → Gemini → GPT-5 → Claude)
and exposes it as an MCP tool using the native MCP Python SDK.
"""

import sys
from pathlib import Path

# Add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import asyncio
import json
import logging
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from config.settings import get_settings
from core.llm_client import (
    CascadeLLMClient,
    DeepSeekAdapter,
    GeminiAdapter,
    GPT5Adapter,
    ClaudeAdapter,
)

# Configure logging to stderr
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Create MCP server
app = Server("llm-cascade")

# Global LLM client
_llm_client: CascadeLLMClient = None


def _initialize_llm_client():
    """Initialize the LLM cascade client."""
    global _llm_client

    if _llm_client is not None:
        return _llm_client

    settings = get_settings()

    # Build adapters in cascade order
    adapters = []
    for llm_name in settings.llm.get_cascade_list():
        if llm_name == "deepseek":
            adapters.append(
                DeepSeekAdapter(
                    model=settings.llm.deepseek_model,
                    api_key=settings.api_keys.deepseek_api_key,
                    base_url=settings.llm.deepseek_base_url,
                    timeout=settings.llm.request_timeout,
                )
            )
        elif llm_name == "gemini":
            adapters.append(
                GeminiAdapter(
                    model=settings.llm.gemini_model,
                    api_key=settings.api_keys.gemini_api_key,
                    timeout=settings.llm.request_timeout,
                )
            )
        elif llm_name in ["gpt5", "gpt4"]:
            adapters.append(
                GPT5Adapter(
                    model=settings.llm.openai_model,
                    api_key=settings.api_keys.openai_api_key,
                    timeout=settings.llm.request_timeout,
                )
            )
        elif llm_name == "claude":
            adapters.append(
                ClaudeAdapter(
                    model=settings.llm.anthropic_model,
                    api_key=settings.api_keys.anthropic_api_key,
                    timeout=settings.llm.request_timeout,
                )
            )

    _llm_client = CascadeLLMClient(
        adapters=adapters,
        max_retries=settings.llm.max_retries,
    )

    logger.info(f"LLM cascade initialized with {len(adapters)} providers")
    return _llm_client


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="generate_completion",
            description="Generate LLM completion using cascade fallback",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "List of conversation messages in OpenAI format",
                    },
                    "tools": {
                        "type": "array",
                        "description": "Optional list of tool schemas available to the LLM",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature (0.0-2.0)",
                        "default": 0.7,
                    },
                },
                "required": ["messages"],
            },
        ),
        Tool(
            name="get_llm_stats",
            description="Get usage statistics for the LLM cascade",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="list_available_providers",
            description="List all available LLM providers in cascade order",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "generate_completion":
            client = _initialize_llm_client()

            messages = arguments.get("messages", [])
            tools = arguments.get("tools")
            temperature = arguments.get("temperature", 0.7)

            response = await client.complete(
                messages=messages,
                tools=tools,
                temperature=temperature,
            )

            result = {
                "content": response.content,
                "tool_calls": response.tool_calls,
                "model": response.model,
                "provider": response.provider,
                "tokens_used": response.tokens_used,
                "finish_reason": response.finish_reason,
            }

            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "get_llm_stats":
            client = _initialize_llm_client()
            result = {
                "usage_stats": client.get_usage_stats(),
                "providers": [adapter.provider_name for adapter in client.adapters],
            }
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "list_available_providers":
            client = _initialize_llm_client()
            providers = [adapter.provider_name for adapter in client.adapters]
            return [TextContent(type="text", text=json.dumps(providers))]

        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
