
# ---------------------------
# FILE: memory.py
# ---------------------------
"""
Simple in-memory session service and memory bank used for the course project.
This is intentionally simple. Replace with persistent stores for production.
"""
from typing import Dict, Any
import threading

class InMemorySessionService:
    def __init__(self):
        self._sessions = {}
        self._lock = threading.Lock()

    def create_session(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            session = {
                'user_id': user_id,
                'history': [],
                'created_at': None,
                'lang': 'en',
            }
            self._sessions[user_id] = session
            return session

    def get_session(self, user_id: str):
        return self._sessions.get(user_id)


class MemoryBank:
    def __init__(self):
        self._store = {}
        self._lock = threading.Lock()

    def save(self, user_id: str, item: Dict[str, Any]):
        with self._lock:
            self._store.setdefault(user_id, []).append(item)

    def get(self, user_id: str):
        return self._store.get(user_id, [])

