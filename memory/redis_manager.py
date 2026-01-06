"""
Redis manager for conversation storage and caching.

Supports both FakeRedis (for development) and real Redis (for production).
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import redis
    import fakeredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from utils.logging import get_logger

logger = get_logger(__name__)


class RedisManager:
    """
    Redis manager for conversation history and caching.

    Automatically uses FakeRedis or real Redis based on configuration.
    """

    def __init__(
        self,
        use_fakeredis: bool = True,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
    ):
        """
        Initialize Redis manager.

        Args:
            use_fakeredis: Use FakeRedis instead of real Redis
            host: Redis host (ignored if use_fakeredis=True)
            port: Redis port (ignored if use_fakeredis=True)
            db: Redis database number
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis and fakeredis packages are required")

        self.use_fakeredis = use_fakeredis

        if use_fakeredis:
            logger.info("Using FakeRedis (in-memory storage)")
            self.client = fakeredis.FakeRedis(decode_responses=True)
        else:
            logger.info(f"Connecting to Redis at {host}:{port}")
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
            )
            # Test connection
            try:
                self.client.ping()
                logger.info("✓ Connected to Redis successfully")
            except redis.ConnectionError as e:
                logger.error(f"✗ Failed to connect to Redis: {e}")
                raise

    def get_conversation(self, session_id: str) -> Dict[str, Any]:
        """
        Get conversation by session ID.

        Args:
            session_id: Session identifier

        Returns:
            Conversation dict with messages and metadata
        """
        key = f"conversation:{session_id}"
        data = self.client.get(key)

        if data is None:
            # Return empty conversation
            return {
                "session_id": session_id,
                "messages": [],
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "metadata": {},
            }

        return json.loads(data)

    def save_conversation(self, session_id: str, conversation: Dict[str, Any]) -> None:
        """
        Save conversation to Redis.

        Args:
            session_id: Session identifier
            conversation: Conversation dict
        """
        key = f"conversation:{session_id}"
        conversation["updated_at"] = datetime.utcnow().isoformat()
        self.client.set(key, json.dumps(conversation))
        logger.debug(f"Saved conversation {session_id}")

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add message to conversation.

        Args:
            session_id: Session identifier
            role: Message role (user, assistant, system, tool)
            content: Message content
            tool_calls: Optional tool calls
            metadata: Optional message metadata
        """
        conversation = self.get_conversation(session_id)

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if tool_calls:
            message["tool_calls"] = tool_calls

        if metadata:
            message["metadata"] = metadata

        conversation["messages"].append(message)
        self.save_conversation(session_id, conversation)

    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        roles: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get messages from conversation.

        Args:
            session_id: Session identifier
            limit: Max number of recent messages to return
            roles: Filter by roles (e.g., ["user", "assistant"])

        Returns:
            List of messages
        """
        conversation = self.get_conversation(session_id)
        messages = conversation["messages"]

        # Filter by roles if specified
        if roles:
            messages = [msg for msg in messages if msg["role"] in roles]

        # Limit to most recent messages
        if limit:
            messages = messages[-limit:]

        return messages

    def clear_conversation(self, session_id: str) -> None:
        """
        Clear conversation history.

        Args:
            session_id: Session identifier
        """
        key = f"conversation:{session_id}"
        self.client.delete(key)
        logger.info(f"Cleared conversation {session_id}")

    def list_sessions(self, pattern: str = "conversation:*") -> List[str]:
        """
        List all session IDs.

        Args:
            pattern: Redis key pattern

        Returns:
            List of session IDs
        """
        keys = self.client.keys(pattern)
        # Extract session IDs from keys
        session_ids = [key.replace("conversation:", "") for key in keys]
        return session_ids

    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """
        Get session metadata.

        Args:
            session_id: Session identifier

        Returns:
            Metadata dict
        """
        conversation = self.get_conversation(session_id)
        return {
            "session_id": session_id,
            "message_count": len(conversation["messages"]),
            "created_at": conversation.get("created_at"),
            "updated_at": conversation.get("updated_at"),
            "metadata": conversation.get("metadata", {}),
        }

    def update_session_metadata(
        self,
        session_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Update session metadata.

        Args:
            session_id: Session identifier
            metadata: Metadata to update/add
        """
        conversation = self.get_conversation(session_id)
        if "metadata" not in conversation:
            conversation["metadata"] = {}
        conversation["metadata"].update(metadata)
        self.save_conversation(session_id, conversation)

    def set_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set cache value.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds (optional)
        """
        cache_key = f"cache:{key}"
        self.client.set(cache_key, json.dumps(value))
        if ttl:
            self.client.expire(cache_key, ttl)

    def get_cache(self, key: str) -> Optional[Any]:
        """
        Get cache value.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        cache_key = f"cache:{key}"
        data = self.client.get(cache_key)
        if data is None:
            return None
        return json.loads(data)

    def delete_cache(self, key: str) -> None:
        """
        Delete cache value.

        Args:
            key: Cache key
        """
        cache_key = f"cache:{key}"
        self.client.delete(cache_key)

    def close(self) -> None:
        """Close Redis connection."""
        if hasattr(self.client, 'close'):
            self.client.close()
            logger.info("Redis connection closed")
