"""
Test MCP orchestrator with tool calls.
"""

import asyncio
from pathlib import Path

from config.settings import get_settings
from core.mcp_orchestrator import MCPOrchestrator
from core.conversation import ConversationManager
from memory.redis_manager import RedisManager
from utils.logging import setup_logger

async def main():
    print("Initializing...\n")

    # Setup
    settings = get_settings()
    setup_logger(level="INFO")

    redis_manager = RedisManager(
        use_fakeredis=True,
        host="localhost",
        port=6379,
        db=0,
    )

    conversation = ConversationManager(
        redis_manager=redis_manager,
        max_history_length=50,
    )

    # Create orchestrator
    orchestrator = MCPOrchestrator(conversation_manager=conversation)

    # Initialize MCP connections
    await orchestrator.initialize()
    print("✓ MCP connections established\n")

    # Test query with tool calls
    test_query = "Load the CSV file at ./data/iris.csv and tell me how many rows it has"
    print(f"Query: {test_query}\n")

    try:
        response = await orchestrator.process_message(
            user_message=test_query,
            session_id="test-session-2",
        )
        print(f"\nResponse: {response}\n")
        print("✓ Test with tool calls successful!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
