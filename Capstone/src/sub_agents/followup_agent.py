"""
Follow-up agent - ADK implementation using long-running operations.
- Schedules reminder using pause()
- Resumes when the reminder event triggers
- Stores follow-up notes in MemoryBank
"""

from adk import Agent, EventActions
from adk.models import ModelMessage
from datetime import datetime, timedelta
from typing import Dict, Any


class FollowUpAgent(Agent):
    def __init__(self, config: Dict[str, Any], memory_bank=None):
        super().__init__(
            name="followup_agent",
            description="Handles follow-up reminders and post-vaccination check-ins.",
            model=config.get("model"),
            instructions=(
                "You schedule follow-up check-ins for the user. "
                "When resumed, send a check-in message and capture any symptoms."
            ),
        )
        self.config = config
        self.memory_bank = memory_bank

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
        resume_at = datetime.utcnow() + timedelta(seconds=delay_seconds)

        # store reminder metadata
        session["followup_resume_at"] = resume_at.isoformat()

        # PAUSE AGENT
        return EventActions(
            pause_until=resume_at,
            message=ModelMessage(
                text=(
                    "Your appointment is confirmed! "
                    "I'll follow up with you shortly after your vaccination."
                )
            ),
        )

    async def _handle_checkin(self, ctx):
        session = ctx.session

        # Form the check-in prompt
        checkin_text = (
            "Hi! How are you feeling after your vaccination? "
            "Any soreness, fever, or other symptoms?"
        )

        # Store that we attempted a follow-up
        if self.memory_bank:
            self.memory_bank.save(
                session["user_id"],
                {"followup_sent_at": datetime.utcnow().isoformat()}
            )

        return ModelMessage(text=checkin_text)
