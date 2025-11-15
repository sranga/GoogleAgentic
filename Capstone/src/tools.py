
# ---------------------------
# FILE: tools.py
# ---------------------------
"""
Custom tools used by agents (file saving, feedback storage, clinic lookup).
"""
import json
from typing import Dict, Any


def save_confirmation_to_file(user_id: str, confirmation: Dict[str, Any]):
    fname = f"confirmation_{user_id}.json"
    with open(fname, 'w') as f:
        json.dump(confirmation, f, indent=2)
    return fname


def store_feedback(memory_bank, user_id: str, feedback: Dict[str, Any]):
    # memory_bank is MemoryBank instance
    memory_bank.save(user_id, feedback)


def clinic_directory_lookup(query: str):
    # placeholder for directory lookup or MCP server call
    return []

