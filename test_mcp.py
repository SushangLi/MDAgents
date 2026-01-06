"""
Quick test script for MCP orchestrator.
"""

import asyncio
from pathlib import Path

from config.settings import get_settings
from core.mcp_orchestrator import MCPOrchestrator
from core.conversation import ConversationManager
from memory.redis_manager import RedisManager
from utils.logging import setup_logger

async def main():
    print("Initializing...")

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

    # Test query
    test_query = "What is 2+2? Just give me a brief answer."
    print(f"Query: {test_query}\n")

    try:
        response = await orchestrator.process_message(
            user_message=test_query,
            session_id="test-session",
        )
        print(f"Response: {response}\n")
        print("✓ Test successful!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up (optional - let it error gracefully)
        try:
            await orchestrator.close()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())
