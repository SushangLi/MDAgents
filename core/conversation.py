"""
Conversation state management.

Handles conversation history, context, and message formatting.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from memory.redis_manager import RedisManager
from utils.logging import get_logger

logger = get_logger(__name__)


class ConversationManager:
    """
    Manages conversation state and history using Redis.

    Handles:
    - Message storage and retrieval
    - Context window management
    - Conversation metadata tracking
    """

    def __init__(
        self,
        redis_manager: RedisManager,
        max_history_length: int = 50,
    ):
        """
        Initialize conversation manager.

        Args:
            redis_manager: Redis manager instance
            max_history_length: Maximum messages to keep in history
        """
        self.redis = redis_manager
        self.max_history_length = max_history_length

    def add_user_message(self, session_id: str, content: str) -> None:
        """
        Add user message to conversation.

        Args:
            session_id: Session identifier
            content: Message content
        """
        self.redis.add_message(
            session_id=session_id,
            role="user",
            content=content,
        )
        logger.debug(f"Added user message to {session_id}")

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add assistant message to conversation.

        Args:
            session_id: Session identifier
            content: Message content
            tool_calls: Optional tool calls made by assistant
            metadata: Optional metadata (e.g., which LLM was used)
        """
        self.redis.add_message(
            session_id=session_id,
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            metadata=metadata,
        )
        logger.debug(f"Added assistant message to {session_id}")

    def add_tool_message(
        self,
        session_id: str,
        tool_name: str,
        tool_result: Any,
    ) -> None:
        """
        Add tool execution result to conversation.

        Args:
            session_id: Session identifier
            tool_name: Name of the tool that was called
            tool_result: Tool execution result
        """
        self.redis.add_message(
            session_id=session_id,
            role="tool",
            content=str(tool_result),
            metadata={"tool_name": tool_name},
        )
        logger.debug(f"Added tool message ({tool_name}) to {session_id}")

    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        include_system: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get conversation messages.

        Args:
            session_id: Session identifier
            limit: Maximum number of recent messages (None for all)
            include_system: Include system message in response

        Returns:
            List of messages formatted for LLM
        """
        # Get messages from Redis
        messages = self.redis.get_messages(
            session_id=session_id,
            limit=limit or self.max_history_length,
        )

        # Format messages for LLM (remove timestamps and internal metadata)
        formatted = []
        for msg in messages:
            formatted_msg = {
                "role": msg["role"],
                "content": msg["content"],
            }
            if "tool_calls" in msg:
                formatted_msg["tool_calls"] = msg["tool_calls"]
            formatted.append(formatted_msg)

        return formatted

    def get_full_context(
        self,
        session_id: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get full conversation context including system prompt.

        Args:
            session_id: Session identifier
            system_prompt: Optional system prompt to prepend

        Returns:
            List of messages with system prompt prepended
        """
        messages = self.get_messages(session_id)

        if system_prompt:
            # Prepend system message
            messages = [{"role": "system", "content": system_prompt}] + messages

        return messages

    def clear_history(self, session_id: str) -> None:
        """
        Clear conversation history.

        Args:
            session_id: Session identifier
        """
        self.redis.clear_conversation(session_id)
        logger.info(f"Cleared history for {session_id}")

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        Get session information.

        Args:
            session_id: Session identifier

        Returns:
            Session metadata
        """
        return self.redis.get_session_metadata(session_id)

    def update_session_metadata(
        self,
        session_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Update session metadata.

        Args:
            session_id: Session identifier
            metadata: Metadata to update
        """
        self.redis.update_session_metadata(session_id, metadata)

    def get_message_count(self, session_id: str) -> int:
        """
        Get number of messages in conversation.

        Args:
            session_id: Session identifier

        Returns:
            Message count
        """
        messages = self.redis.get_messages(session_id)
        return len(messages)

    def needs_summarization(self, session_id: str) -> bool:
        """
        Check if conversation needs summarization.

        Args:
            session_id: Session identifier

        Returns:
            True if conversation exceeds max length
        """
        count = self.get_message_count(session_id)
        return count > self.max_history_length

    def get_recent_exchanges(
        self,
        session_id: str,
        num_exchanges: int = 5,
    ) -> List[tuple]:
        """
        Get recent user-assistant exchanges.

        Args:
            session_id: Session identifier
            num_exchanges: Number of exchanges to retrieve

        Returns:
            List of (user_message, assistant_message) tuples
        """
        messages = self.redis.get_messages(session_id)

        exchanges = []
        user_msg = None

        for msg in reversed(messages):
            if msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant" and user_msg:
                exchanges.insert(0, (user_msg, msg["content"]))
                user_msg = None
                if len(exchanges) >= num_exchanges:
                    break

        return exchanges
