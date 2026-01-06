"""
Main entry point for MCP-Native Multi-Agent Scientific Computing System.

Everything communicates via Model Context Protocol (MCP):
- LLM accessed via MCP
- Tools accessed via MCP
- Agents accessed via MCP

The orchestrator is a thin MCP client that routes messages.
"""

# Suppress warnings before any imports
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import asyncio
import sys
import uuid
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from config.settings import get_settings
from core.mcp_orchestrator import MCPOrchestrator
from core.conversation import ConversationManager
from memory.redis_manager import RedisManager
from utils.logging import setup_logger
from utils.prompts import get_welcome_message, get_help_message

logger = setup_logger()


class MCPInteractiveCLI:
    """Interactive CLI for MCP-native multi-agent system."""

    def __init__(self):
        """Initialize the CLI."""
        self.console = Console()
        self.session_id = str(uuid.uuid4())
        self.settings = None
        self.orchestrator = None
        self.conversation = None

    async def initialize(self):
        """Initialize all components."""
        self.console.print("[yellow]Initializing MCP-native system...[/yellow]")

        try:
            # Load settings
            self.settings = get_settings()

            # Initialize Redis manager
            redis_manager = RedisManager(
                use_fakeredis=self.settings.redis.redis_use_fakeredis,
                host=self.settings.redis.redis_host,
                port=self.settings.redis.redis_port,
                db=self.settings.redis.redis_db,
            )

            # Initialize conversation manager
            self.conversation = ConversationManager(
                redis_manager=redis_manager,
                max_history_length=self.settings.app.max_conversation_length,
            )

            # Initialize MCP orchestrator
            self.console.print("[yellow]Starting MCP servers...[/yellow]")
            self.orchestrator = MCPOrchestrator(
                conversation_manager=self.conversation,
            )

            # Connect to all MCP servers
            await self.orchestrator.initialize()

            self.console.print("[green]✓ MCP-native system initialized![/green]")
            self.console.print("[cyan]Architecture: All communication via MCP protocol[/cyan]\n")

        except Exception as e:
            self.console.print(f"[red]✗ Initialization failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            raise

    async def run(self):
        """Run the interactive REPL loop."""
        # Display welcome message
        welcome = get_welcome_message()
        welcome += "\n\n[bold cyan]🔗 MCP-Native Architecture[/bold cyan]\n"
        welcome += "• LLM → MCP Server\n"
        welcome += "• Tools → MCP Server\n"
        welcome += "• Agents → MCP Server\n"
        welcome += "• All communication via MCP protocol\n"

        self.console.print(Panel(welcome, style="bold blue"))

        while True:
            try:
                # Get user input
                user_input = self.console.input("\n[bold blue]You:[/bold blue] ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.startswith("/"):
                    if not await self.handle_command(user_input):
                        break  # Exit requested
                    continue

                # Process message via MCP
                with self.console.status("[bold yellow]Processing via MCP...[/bold yellow]"):
                    response = await self.orchestrator.process_message(
                        user_message=user_input,
                        session_id=self.session_id,
                    )

                # Display response
                self.console.print(f"\n[bold green]Assistant:[/bold green]")
                self.console.print(Markdown(response))

            except KeyboardInterrupt:
                self.console.print("\n\n[yellow]Interrupted. Use /exit to quit.[/yellow]")
                continue
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"\n[red]✗ Error: {e}[/red]")
                import traceback
                traceback.print_exc()
                continue

    async def handle_command(self, command: str) -> bool:
        """
        Handle special commands.

        Args:
            command: Command string

        Returns:
            False if exit requested, True otherwise
        """
        cmd = command.lower().strip()

        if cmd == "/exit" or cmd == "/quit":
            self.console.print("\n[yellow]Closing MCP connections...[/yellow]")
            await self.orchestrator.close()
            self.console.print("[yellow]Goodbye![/yellow]")
            return False

        elif cmd == "/help":
            help_msg = get_help_message()
            help_msg += "\n\n[bold cyan]MCP Architecture:[/bold cyan]\n"
            help_msg += "This system uses pure MCP protocol for all communications.\n"
            help_msg += "The LLM drives the workflow via tool calls returned through MCP.\n"
            self.console.print(Panel(help_msg, title="Help", style="cyan"))

        elif cmd == "/clear":
            self.conversation.clear_history(self.session_id)
            self.console.print("[green]✓ Conversation history cleared[/green]")

        elif cmd == "/new":
            self.session_id = str(uuid.uuid4())
            self.console.print(f"[green]✓ Started new session: {self.session_id[:8]}...[/green]")

        elif cmd == "/stats":
            stats = self.orchestrator.get_usage_stats()
            self.console.print(Panel(
                f"MCP Servers:\n"
                f"  • LLM: {stats['mcp_servers']['llm']}\n"
                f"  • Tools: {stats['mcp_servers']['tools']}\n"
                f"  • Agents: {stats['mcp_servers']['agents']}\n\n"
                f"Session: {self.session_id[:8]}...\n"
                f"Messages: {self.conversation.get_message_count(self.session_id)}",
                title="MCP Statistics",
                style="cyan"
            ))

        elif cmd == "/mcp":
            # Show MCP server info
            self.console.print(Panel(
                "[bold]MCP Server Architecture:[/bold]\n\n"
                "1. [cyan]LLM MCP Server[/cyan]\n"
                "   • Wraps LLM cascade (DeepSeek → Gemini → GPT-5 → Claude)\n"
                "   • Tool: generate_completion(messages, tools)\n\n"
                "2. [cyan]Tools MCP Server[/cyan]\n"
                "   • File operations (read, write, list, search)\n"
                "   • Data analysis (load_csv, analyze, plot)\n"
                "   • Visualization (matplotlib, plotly, seaborn)\n\n"
                "3. [cyan]Agents MCP Server[/cyan]\n"
                "   • nanobanana: Image generation (Gemini Flash)\n"
                "   • create_cover_image, create_abstract_figure\n\n"
                "[bold yellow]All communication happens via MCP protocol![/bold yellow]",
                title="MCP Architecture",
                style="blue"
            ))

        else:
            self.console.print(f"[red]Unknown command: {command}[/red]")
            self.console.print("[yellow]Type /help for available commands[/yellow]")

        return True

    async def cleanup(self):
        """Cleanup resources."""
        if self.orchestrator:
            await self.orchestrator.close()


async def main():
    """Main entry point."""
    cli = MCPInteractiveCLI()

    try:
        await cli.initialize()
        await cli.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await cli.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
