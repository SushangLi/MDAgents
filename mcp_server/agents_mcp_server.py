"""
Agents MCP Server - Exposes expert agents via Model Context Protocol.

This server wraps expert agents (like nanobanana) using the native MCP Python SDK.
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

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from config.settings import get_settings
from agents.nanobanana_agent import NanobananaAgent

# Configure logging to stderr
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Create MCP server
app = Server("expert-agents")

# Global agent instances
_nanobanana: NanobananaAgent = None


def _initialize_agents():
    """Initialize expert agents."""
    global _nanobanana

    if _nanobanana is not None:
        return

    settings = get_settings()

    _nanobanana = NanobananaAgent(
        api_key=settings.api_keys.gemini_api_key,
        model=settings.llm.gemini_model,
        output_dir=str(settings.app.plots_dir),
    )

    logger.info("Expert agents initialized")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="create_cover_image",
            description="Generate a scientific article cover image using nanobanana agent",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Description of the image to generate",
                    },
                    "style": {
                        "type": "string",
                        "description": "Visual style ('scientific', 'abstract', 'technical')",
                        "default": "scientific",
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="create_abstract_figure",
            description="Create a graphical abstract for a scientific paper using nanobanana agent",
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Description of the abstract content",
                    },
                    "style": {
                        "type": "string",
                        "description": "Visual style preference",
                        "default": "clean",
                    },
                },
                "required": ["description"],
            },
        ),
        Tool(
            name="process_image",
            description="Process an existing image with specific instructions using nanobanana agent",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to input image",
                    },
                    "instruction": {
                        "type": "string",
                        "description": "Processing instructions",
                    },
                },
                "required": ["image_path", "instruction"],
            },
        ),
        Tool(
            name="list_available_agents",
            description="List all available expert agents",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    _initialize_agents()

    try:
        if name == "create_cover_image":
            file_path = await _nanobanana.create_cover_image(
                prompt=arguments["prompt"],
                style=arguments.get("style", "scientific"),
            )
            result = {
                "success": True,
                "file_path": file_path,
                "agent": "nanobanana",
            }
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "create_abstract_figure":
            file_path = await _nanobanana.create_abstract_figure(
                description=arguments["description"],
                style=arguments.get("style", "clean"),
            )
            result = {
                "success": True,
                "file_path": file_path,
                "agent": "nanobanana",
            }
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "process_image":
            file_path = await _nanobanana.process_image(
                image_path=arguments["image_path"],
                instruction=arguments["instruction"],
            )
            result = {
                "success": True,
                "file_path": file_path,
                "agent": "nanobanana",
            }
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "list_available_agents":
            result = [
                {
                    "name": "nanobanana",
                    "description": "Gemini Flash-powered image generation agent",
                    "capabilities": [
                        "create_cover_image",
                        "create_abstract_figure",
                        "process_image",
                    ],
                }
            ]
            return [TextContent(type="text", text=json.dumps(result))]

        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        result = {"success": False, "error": str(e)}
        return [TextContent(type="text", text=json.dumps(result))]


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
