"""
Main entry point for the Multi-Agent Scientific Computing System.

Provides an interactive CLI for data analysis, visualization, and
scientific computing tasks.
"""

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
from utils.logging import setup_logger, console as rich_console
from utils.prompts import get_welcome_message, get_help_message


class InteractiveCLI:
    """Interactive command-line interface for the multi-agent system."""

    def __init__(self):
        """Initialize the CLI."""
        self.console = Console()
        self.session_id = str(uuid.uuid4())
        self.settings = None
        self.orchestrator = None
        self.conversation = None

    async def initialize(self):
        """Initialize all components."""
        self.console.print("[yellow]Initializing system...[/yellow]")

        try:
            # Load settings
            self.settings = get_settings()

            # Setup logging
            setup_logger(
                level=self.settings.app.log_level,
                log_file=Path("./logs/multiagents.log") if self.settings.app.log_level == "DEBUG" else None,
            )

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
            self.orchestrator = MCPOrchestrator(
                conversation_manager=self.conversation,
            )

            # Initialize MCP connections (connects to all MCP servers)
            await self.orchestrator.initialize()

            self.console.print("[green]✓ System initialized successfully![/green]\n")

        except Exception as e:
            self.console.print(f"[red]✗ Initialization failed: {e}[/red]")
            raise

    async def run(self):
        """Run the interactive REPL loop."""
        # Display welcome message
        self.console.print(Panel(get_welcome_message(), style="bold blue"))

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

                # Process message
                with self.console.status("[bold yellow]Thinking...[/bold yellow]"):
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
            self.console.print("\n[yellow]Goodbye![/yellow]")
            return False

        elif cmd == "/help":
            self.console.print(Panel(get_help_message(), title="Help", style="cyan"))

        elif cmd == "/clear":
            self.conversation.clear_history(self.session_id)
            self.console.print("[green]✓ Conversation history cleared[/green]")

        elif cmd == "/new":
            self.session_id = str(uuid.uuid4())
            self.console.print(f"[green]✓ Started new session: {self.session_id[:8]}...[/green]")

        elif cmd == "/stats":
            stats = self.orchestrator.get_usage_stats()
            mcp_servers = stats.get("mcp_servers", {})
            self.console.print(Panel(
                f"MCP Servers:\n"
                f"  LLM: {mcp_servers.get('llm', 'unknown')}\n"
                f"  Tools: {mcp_servers.get('tools', 'unknown')}\n"
                f"  Agents: {mcp_servers.get('agents', 'unknown')}\n"
                f"\nSession: {self.session_id[:8]}...\n"
                f"Messages: {self.conversation.get_message_count(self.session_id)}",
                title="Statistics",
                style="cyan"
            ))

        else:
            self.console.print(f"[red]Unknown command: {command}[/red]")
            self.console.print("[yellow]Type /help for available commands[/yellow]")

        return True


async def main():
    """Main entry point."""
    cli = InteractiveCLI()

    try:
        await cli.initialize()
        await cli.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
    finally:
        # Clean up MCP connections
        if cli.orchestrator:
            await cli.orchestrator.close()


if __name__ == "__main__":
    asyncio.run(main())
