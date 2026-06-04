# src/enterprise_ai_cache/cache/message_history.py
"""Semantic message history for conversation management using Redis Stack."""

import json
from typing import Optional, Union, List, Dict, Any, Callable
from enum import Enum
import numpy as np
import redis
from enterprise_ai_cache.config.settings import Settings, get_settings
from enterprise_ai_cache.exceptions import CacheError, ConnectionError, ValidationError


class MessageRole(str, Enum):
    """Standard message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class SemanticMessageHistory:
    """Manages conversation history per session with relevance search.

    Supports role-based filtering, relevance search, and session isolation.
    """

    HISTORY_SUFFIX = ":history"
    INDEX_SUFFIX = ":history_idx"

    def __init__(
        self,
        name: str,
        ttl: int = 86400,
        settings: Optional[Settings] = None,
        redis_url: Optional[str] = None,
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        embedding_method: Optional[Callable[[str], np.ndarray]] = None,
    ):
        """Initialize the message history manager.

        Args:
            name: Session name (used as key suffix).
            ttl: Time-to-live for history entries in seconds.
            settings: Configuration settings.
            redis_url: Redis host address.
            redis_port: Redis port.
            redis_password: Redis password.
            embedding_method: Optional function to embed messages for semantic search.
        """
        self.name = name
        self.ttl = ttl
        self.embedding_method = embedding_method
        self.settings = settings or get_settings()

        try:
            self._redis = redis.Redis(
                host=redis_url or self.settings.redis.host,
                port=redis_port,
                password=redis_password or self.settings.redis.password,
                db=self.settings.redis.db,
                decode_responses=True,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")

        self._history_key = f"{name}{self.HISTORY_SUFFIX}"
        self._index_name = f"{name}{self.INDEX_SUFFIX}"

    def get_history(self) -> List[Dict[str, Any]]:
        """Get all conversation history.

        Returns:
            List of message dictionaries.
        """
        try:
            data = self._redis.get(self._history_key)
            if not data:
                return []
            return json.loads(data)
        except json.JSONDecodeError:
            return []
        except Exception as e:
            raise CacheError(f"Failed to get history: {e}")

    def add_message(self, message: Union[Dict[str, Any], List[Dict[str, Any]]]) -> bool:
        """Add message(s) to the conversation history.

        Args:
            message: Single message or list of messages.
                    Each message should have 'role' and 'content' keys.

        Returns:
            True if added successfully.
        """
        if isinstance(message, dict):
            message = [message]

        self._validate_messages(message)

        try:
            history = self.get_history()
            history.extend(message)
            self._redis.setex(self._history_key, self.ttl, json.dumps(history, ensure_ascii=False))
            return True
        except Exception as e:
            raise CacheError(f"Failed to add message: {e}")

    def _validate_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Validate message format.

        Args:
            messages: List of messages to validate.

        Raises:
            ValidationError: If messages are not properly formatted.
        """
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValidationError("Each message must be a dictionary")
            if "role" not in msg:
                raise ValidationError("Message must have a 'role' field")
            if "content" not in msg:
                raise ValidationError("Message must have a 'content' field")

    def get_recent(self, role: Optional[Union[str, List[str]]] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages from the conversation.

        Args:
            role: Filter by role(s) (e.g., 'user', 'assistant').
            top_k: Maximum number of messages to return.

        Returns:
            List of recent messages.
        """
        history = self.get_history()

        if role:
            if isinstance(role, str):
                role = [role]
            history = [m for m in history if m.get("role", "") in role]

        return history[-top_k:] if top_k else history

    def get_relevant(self, content: str, top_k: int = 10, case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """Get messages relevant to a query string.

        Args:
            content: Query string to search for.
            top_k: Maximum number of messages to return.
            case_sensitive: Whether to perform case-sensitive search.

        Returns:
            List of relevant messages.
        """
        history = self.get_history()

        if case_sensitive:
            relevant = [m for m in history if content in m.get("content", "")]
        else:
            content_lower = content.lower()
            relevant = [m for m in history if content_lower in m.get("content", "").lower()]

        return relevant[:top_k]

    def get_by_role(self, role: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all messages with a specific role.

        Args:
            role: Role to filter by.
            top_k: Optional limit on number of messages.

        Returns:
            List of messages with the specified role.
        """
        history = self.get_history()
        filtered = [m for m in history if m.get("role", "") == role]
        return filtered[-top_k:] if top_k else filtered

    def delete_history(self, top_k: Optional[int] = None) -> bool:
        """Delete recent messages from history.

        Args:
            top_k: Number of recent messages to delete. If None, deletes all.

        Returns:
            True if deleted successfully.
        """
        try:
            history = self.get_history()
            if top_k is None:
                history = []
            else:
                history = history[:-top_k] if top_k else history
            self._redis.setex(self._history_key, self.ttl, json.dumps(history, ensure_ascii=False))
            return True
        except Exception as e:
            raise CacheError(f"Failed to delete history: {e}")

    def clear_history(self) -> bool:
        """Clear all conversation history for this session.

        Returns:
            True if cleared successfully.
        """
        try:
            self._redis.delete(self._history_key)
            return True
        except Exception as e:
            raise CacheError(f"Failed to clear history: {e}")

    def count(self) -> int:
        """Get the number of messages in history.

        Returns:
            Number of messages.
        """
        history = self.get_history()
        return len(history)

    def count_by_role(self, role: str) -> int:
        """Count messages by a specific role.

        Args:
            role: Role to count.

        Returns:
            Number of messages with the specified role.
        """
        history = self.get_history()
        return len([m for m in history if m.get("role", "") == role])

    def export(self) -> str:
        """Export history as JSON string.

        Returns:
            JSON string of all messages.
        """
        history = self.get_history()
        return json.dumps(history, ensure_ascii=False, indent=2)

    def import_history(self, data: Union[str, List[Dict[str, Any]]]) -> bool:
        """Import history from JSON string or list.

        Args:
            data: JSON string or list of messages.

        Returns:
            True if imported successfully.
        """
        try:
            if isinstance(data, str):
                messages = json.loads(data)
            else:
                messages = data

            if not isinstance(messages, list):
                raise ValidationError("Import data must be a list of messages")

            self._validate_messages(messages)
            self._redis.setex(self._history_key, self.ttl, json.dumps(messages, ensure_ascii=False))
            return True
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {e}")
        except Exception as e:
            raise CacheError(f"Failed to import history: {e}")

    def stats(self) -> Dict[str, Any]:
        """Get history statistics.

        Returns:
            Dictionary with history statistics.
        """
        history = self.get_history()
        roles = {}
        for msg in history:
            role = msg.get("role", "unknown")
            roles[role] = roles.get(role, 0) + 1

        return {
            "name": self.name,
            "total_messages": len(history),
            "roles": roles,
            "ttl": self.ttl,
        }
