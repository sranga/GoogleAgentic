"""
Memory Services

Provides session management and persistent memory storage for the V-Access system.

Components:
- InMemorySessionService: Manages user sessions during conversations
- MemoryBank: Stores user-specific data across sessions

For production, replace with persistent stores (Redis, PostgreSQL, etc.)
"""

import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, UTC


class InMemorySessionService:
    """
    Manages user sessions for active conversations.
    Thread-safe for concurrent access.

    In production, replace with Redis or similar distributed session store.
    """

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_session(self, user_id: str) -> Dict[str, Any]:
        """
        Create a new session for a user.

        Args:
            user_id: Unique user identifier

        Returns:
            New session dictionary
        """
        with self._lock:
            session = {
                "user_id": user_id,
                "history": [],
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "lang": "en",
            }
            self._sessions[user_id] = session
            return session

    def get_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an existing session.

        Args:
            user_id: User identifier

        Returns:
            Session dictionary or None if not found
        """
        with self._lock:
            return self._sessions.get(user_id)

    def update_session(self, user_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update an existing session.

        Args:
            user_id: User identifier
            updates: Dictionary of fields to update

        Returns:
            Updated session or None if not found
        """
        with self._lock:
            session = self._sessions.get(user_id)
            if session:
                session.update(updates)
                session["updated_at"] = datetime.now(UTC).isoformat()
            return session

    def delete_session(self, user_id: str) -> bool:
        """
        Delete a session.

        Args:
            user_id: User identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if user_id in self._sessions:
                del self._sessions[user_id]
                return True
            return False

    def get_active_session_count(self) -> int:
        """Return count of active sessions."""
        with self._lock:
            return len(self._sessions)

    def cleanup_stale_sessions(self, max_age_hours: int = 24):
        """
        Remove sessions older than max_age_hours.

        Args:
            max_age_hours: Maximum session age in hours
        """
        from datetime import timedelta

        cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)

        with self._lock:
            stale_users = [
                user_id for user_id, session in self._sessions.items()
                if datetime.fromisoformat(session["created_at"]) < cutoff
            ]
            for user_id in stale_users:
                del self._sessions[user_id]

            return len(stale_users)


class MemoryBank:
    """
    Long-term memory storage for user-specific data.
    Persists across sessions for personalization and analytics.

    In production, replace with a database (PostgreSQL, MongoDB, etc.)
    """

    def __init__(self):
        self._store: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def save(self, user_id: str, item: Dict[str, Any]):
        """
        Save an item to user's memory.

        Args:
            user_id: User identifier
            item: Data to store
        """
        with self._lock:
            if user_id not in self._store:
                self._store[user_id] = []

            # Add metadata
            item_with_meta = item.copy()
            item_with_meta["_saved_at"] = datetime.now(UTC).isoformat()

            self._store[user_id].append(item_with_meta)

    def get(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all items for a user.

        Args:
            user_id: User identifier

        Returns:
            List of stored items
        """
        with self._lock:
            return self._store.get(user_id, []).copy()

    def get_latest(self, user_id: str, event_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the most recent item for a user.

        Args:
            user_id: User identifier
            event_type: Optional filter by event type

        Returns:
            Most recent item or None
        """
        items = self.get(user_id)
        if event_type:
            items = [i for i in items if i.get("event") == event_type]

        return items[-1] if items else None

    def delete(self, user_id: str) -> bool:
        """
        Delete all items for a user.

        Args:
            user_id: User identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if user_id in self._store:
                del self._store[user_id]
                return True
            return False

    def get_user_count(self) -> int:
        """Return count of users with stored data."""
        with self._lock:
            return len(self._store)

    def get_total_items(self) -> int:
        """Return total count of all stored items."""
        with self._lock:
            return sum(len(items) for items in self._store.values())