"""
Follow-up agent - ADK implementation using long-running operations.
- Schedules reminder using pause()
- Resumes when the reminder event triggers
- Stores follow-up notes in MemoryBank
"""
from google.genai import types
from pydantic import PrivateAttr

# Use the try-except block to handle testing environment
try:
    from google.adk import Agent
    from google.adk.events import EventActions
except ImportError:
    # Define simple mock classes for testing
    class Agent:
        def __init__(self, *args, **kwargs):
            pass

    class EventActions:
        def __init__(self, *args, **kwargs):
            pass

from datetime import datetime, timedelta, UTC
from typing import Dict, Any


class FollowUpAgent(Agent):

    # Declare private attributes using PrivateAttr
    _memory_bank: Any = PrivateAttr(default=None)
    _config: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, config: Dict[str, Any], memory_bank=None):
        super().__init__(
            name="followup_agent",
            description="Handles follow-up reminders and post-vaccination check-ins."
                        "You schedule follow-up check-ins for the user. "
                        "When resumed, send a check-in message and capture any symptoms.",
            model=config.get("model"),
        )
        object.__setattr__(self, '_config', config)
        object.__setattr__(self, '_memory_bank', memory_bank)

    @property
    def config(self):
        return self._config

    @property
    def memory_bank(self):
        return self._memory_bank

    async def on_event(self, event, ctx):
        session = ctx.session

        # 1. Check if we are resuming after pause()
        if event.resume:
            # This is the follow-up event firing
            return await self._handle_checkin(ctx)

        # 2. Otherwise, this is the initial call to schedule a reminder
        return await self._schedule_reminder(ctx)

    async def _schedule_reminder(self, ctx):
        session = ctx.session

        # compute when to resume
        delay_seconds = self.config.get("followup_seconds", 5)
        resume_at = datetime.now(UTC) + timedelta(seconds=delay_seconds)

        # store reminder metadata
        session["followup_resume_at"] = resume_at.isoformat()

        # PAUSE AGENT
        return EventActions(
            state_delta={"pause_until":resume_at,
            "message":"Your appointment is confirmed! I'll follow up with you shortly after your vaccination."}
        )

    async def _handle_checkin(self, ctx):
        session = ctx.session

        # Form the check-in prompt
        checkin_text = "Hi! How are you feeling after your vaccination? Any soreness, fever, or other symptoms?"

        # Store that we attempted a follow-up
        if self.memory_bank:
            self.memory_bank.save(
                session["user_id"],
                {"followup_sent_at": datetime.now(UTC).isoformat()}
            )

        return EventActions(state_delta={"message": checkin_text})

    async def _mock_call_tool(self, tool_name: str, params: dict):
        """Mock tool call (not used by this agent)."""
        return {}

    async def emit(self, payload: dict, session: dict):
        """Convenience method for testing and direct invocation."""
        fake_event = type("Event", (), {"payload": payload, "resume": False})()
        fake_ctx = type("Context", (), {
            "session": session,
            "metrics": None,
            "call_tool": self._mock_call_tool  # Keep for compatibility
        })()

        response = await self.on_event(fake_event, fake_ctx)
        return response