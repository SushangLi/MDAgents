"""
Unified MCP Server exposing file, data, and plotting tools.

Uses the native MCP Python SDK for guaranteed stdio compatibility.
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

import tools.file_tools as file_tools
import tools.data_tools as data_tools
import tools.plot_tools as plot_tools

# Configure logging to stderr
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Create MCP server
app = Server("scientific-computing-tools")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        # File operations
        Tool(
            name="read_file",
            description="Read file contents from disk",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="write_file",
            description="Write content to a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write"},
                    "content": {"type": "string", "description": "Content to write to the file"},
                },
                "required": ["path", "content"],
            },
        ),
        Tool(
            name="list_directory",
            description="List contents of a directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the directory"},
                    "pattern": {"type": "string", "description": "Optional glob pattern to filter files", "default": "*"},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="search_files",
            description="Search for files matching a pattern recursively",
            inputSchema={
                "type": "object",
                "properties": {
                    "root_dir": {"type": "string", "description": "Root directory to search from"},
                    "pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.py', '*.csv')"},
                },
                "required": ["root_dir", "pattern"],
            },
        ),
        # Data analysis
        Tool(
            name="load_csv",
            description="Load CSV file into a DataFrame for analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the CSV file"},
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="load_excel",
            description="Load Excel file into a DataFrame for analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the Excel file"},
                    "sheet_name": {"type": "string", "description": "Optional sheet name"},
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="analyze_dataframe",
            description="Analyze DataFrame and return comprehensive statistics",
            inputSchema={
                "type": "object",
                "properties": {
                    "df_id": {"type": "string", "description": "DataFrame identifier from load_csv or load_excel"},
                },
                "required": ["df_id"],
            },
        ),
        Tool(
            name="get_dataframe_head",
            description="Get first n rows of a DataFrame",
            inputSchema={
                "type": "object",
                "properties": {
                    "df_id": {"type": "string", "description": "DataFrame identifier"},
                    "n": {"type": "integer", "description": "Number of rows to return", "default": 10},
                },
                "required": ["df_id"],
            },
        ),
        Tool(
            name="query_dataframe",
            description="Execute pandas query on a DataFrame",
            inputSchema={
                "type": "object",
                "properties": {
                    "df_id": {"type": "string", "description": "DataFrame identifier"},
                    "query": {"type": "string", "description": "Pandas query string (e.g., 'column > 10')"},
                },
                "required": ["df_id", "query"],
            },
        ),
        Tool(
            name="compute_statistics",
            description="Compute statistics for specified columns",
            inputSchema={
                "type": "object",
                "properties": {
                    "df_id": {"type": "string", "description": "DataFrame identifier"},
                    "columns": {"type": "array", "description": "List of column names (None for all numeric columns)"},
                },
                "required": ["df_id"],
            },
        ),
        # Visualization
        Tool(
            name="create_matplotlib_plot",
            description="Create a matplotlib plot from DataFrame data",
            inputSchema={
                "type": "object",
                "properties": {
                    "df_id": {"type": "string", "description": "DataFrame identifier"},
                    "plot_type": {"type": "string", "description": "Type of plot (line, scatter, bar, hist, box)"},
                    "x": {"type": "string", "description": "X-axis column name"},
                    "y": {"type": "string", "description": "Y-axis column name"},
                    "title": {"type": "string", "description": "Plot title"},
                    "xlabel": {"type": "string", "description": "X-axis label"},
                    "ylabel": {"type": "string", "description": "Y-axis label"},
                },
                "required": ["df_id", "plot_type"],
            },
        ),
        Tool(
            name="create_plotly_plot",
            description="Create an interactive plotly plot from DataFrame data",
            inputSchema={
                "type": "object",
                "properties": {
                    "df_id": {"type": "string", "description": "DataFrame identifier"},
                    "plot_type": {"type": "string", "description": "Type of plot (line, scatter, bar, histogram, box)"},
                    "x": {"type": "string", "description": "X-axis column name"},
                    "y": {"type": "string", "description": "Y-axis column name"},
                    "title": {"type": "string", "description": "Plot title"},
                },
                "required": ["df_id", "plot_type"],
            },
        ),
        Tool(
            name="create_seaborn_plot",
            description="Create a seaborn plot from DataFrame data",
            inputSchema={
                "type": "object",
                "properties": {
                    "df_id": {"type": "string", "description": "DataFrame identifier"},
                    "plot_type": {"type": "string", "description": "Type of plot (scatter, line, bar, box, violin, heatmap)"},
                    "x": {"type": "string", "description": "X-axis column name"},
                    "y": {"type": "string", "description": "Y-axis column name"},
                    "hue": {"type": "string", "description": "Grouping variable for color coding"},
                    "title": {"type": "string", "description": "Plot title"},
                },
                "required": ["df_id", "plot_type"],
            },
        ),
        Tool(
            name="save_figure",
            description="Save a figure to a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "fig_id": {"type": "string", "description": "Figure identifier from create_*_plot functions"},
                    "file_path": {"type": "string", "description": "Output file path (supports .png, .jpg, .pdf, .svg, .html)"},
                },
                "required": ["fig_id", "file_path"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        # File operations
        if name == "read_file":
            result = file_tools.read_file(arguments["path"])
            return [TextContent(type="text", text=result)]

        elif name == "write_file":
            success = file_tools.write_file(arguments["path"], arguments["content"])
            result = {"success": success, "path": arguments["path"]}
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "list_directory":
            pattern = arguments.get("pattern", "*")
            result = file_tools.list_directory(arguments["path"], None if pattern == "*" else pattern)
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "search_files":
            result = file_tools.search_files(arguments["root_dir"], arguments["pattern"])
            return [TextContent(type="text", text=json.dumps(result))]

        # Data analysis
        elif name == "load_csv":
            result = data_tools.load_csv(arguments["file_path"])
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "load_excel":
            sheet_name = arguments.get("sheet_name")
            result = data_tools.load_excel(arguments["file_path"], sheet_name)
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "analyze_dataframe":
            result = data_tools.analyze_dataframe(arguments["df_id"])
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "get_dataframe_head":
            n = arguments.get("n", 10)
            result = data_tools.get_dataframe_head(arguments["df_id"], n)
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "query_dataframe":
            result = data_tools.query_dataframe(arguments["df_id"], arguments["query"])
            return [TextContent(type="text", text=result)]

        elif name == "compute_statistics":
            columns = arguments.get("columns")
            result = data_tools.compute_statistics(arguments["df_id"], columns)
            return [TextContent(type="text", text=json.dumps(result))]

        # Visualization
        elif name == "create_matplotlib_plot":
            fig_id = plot_tools.create_matplotlib_plot(
                df_id=arguments["df_id"],
                plot_type=arguments["plot_type"],
                x=arguments.get("x"),
                y=arguments.get("y"),
                title=arguments.get("title"),
                xlabel=arguments.get("xlabel"),
                ylabel=arguments.get("ylabel"),
            )
            return [TextContent(type="text", text=fig_id)]

        elif name == "create_plotly_plot":
            fig_id = plot_tools.create_plotly_plot(
                df_id=arguments["df_id"],
                plot_type=arguments["plot_type"],
                x=arguments.get("x"),
                y=arguments.get("y"),
                title=arguments.get("title"),
            )
            return [TextContent(type="text", text=fig_id)]

        elif name == "create_seaborn_plot":
            fig_id = plot_tools.create_seaborn_plot(
                df_id=arguments["df_id"],
                plot_type=arguments["plot_type"],
                x=arguments.get("x"),
                y=arguments.get("y"),
                hue=arguments.get("hue"),
                title=arguments.get("title"),
            )
            return [TextContent(type="text", text=fig_id)]

        elif name == "save_figure":
            file_path = plot_tools.save_figure(arguments["fig_id"], arguments["file_path"])
            return [TextContent(type="text", text=file_path)]

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
