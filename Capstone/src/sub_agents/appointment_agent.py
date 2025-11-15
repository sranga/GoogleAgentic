
# ---------------------------
# FILE: sub_agents/appointment_agent.py
# ---------------------------
"""
Appointment agent
- Sequential tool-calling
- OpenAPI booking tool usage
- Slot validation
- LoopAgent-style retry logic
"""
from typing import Dict, Any, List
from adk import Agent, EventActions
from adk.models import ModelMessage
from adk.tools import OpenAPITool

# Mock OpenAPI schema
BOOKING_OPENAPI_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Clinic Booking API", "version": "1.0.0"},
    "paths": {
        "/book": {
            "post": {
                "summary": "Book appointment",
                "requestBody": {"required": True},
                "responses": {
                    "200": {
                        "description": "Success",
                        "content": {"application/json": {"example": {"confirmation_id": "CONF-001"}}}
                    }
                }
            }
        }
    }
}

booking_tool = OpenAPITool(
    name="booking_api",
    description="Schedules vaccine appointments via clinic API",
    specification=BOOKING_OPENAPI_SPEC,
)


class AppointmentAgent(Agent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="appointment_agent",
            model=config.get("model"),
            description="Chooses the best clinic slot and books an appointment.",
            instructions=(
                "You book appointments using the booking_api tool. "
                "Never fabricate confirmation IDs — always return tool results. "
                "If slot selection fails, restart the event."
            ),
            tools=[booking_tool],
        )
        self.max_retries = 3

    async def on_event(self, event, ctx):
        session = ctx.session
        clinics: List[Dict[str, Any]] = session.get("last_clinics", [])
        attempt = session.get("appointment_attempt", 1)

        # If already retried max times
        if attempt > self.max_retries:
            return ModelMessage(
                text="I wasn’t able to confirm an appointment after several attempts. "
                     "Please try again or pick another clinic."
            )

        # Step 1: pick a slot
        slot = self._choose_slot(clinics)
        if not self._validate_slot(slot):
            session["appointment_attempt"] = attempt + 1
            return EventActions(
                restart=True,
                message=ModelMessage(text="Retrying — slot wasn’t valid, attempting again...")
            )

        # Step 2: call booking API tool
        tool_args = {
            "clinic_id": slot["clinic_id"],
            "time": slot["time"],
            "user_id": session["user_id"],
        }

        tool_result = await ctx.call_tool("booking_api", tool_args)

        # Reset retry counter after success
        session["appointment_attempt"] = 1

        return tool_result

    def _choose_slot(self, clinics: List[Dict[str, Any]]):
        for c in clinics:
            if c.get("has_api"):
                return {"clinic_id": c["id"], "time": "2025-12-01T10:00:00Z"}
        return None

    def _validate_slot(self, slot: Dict[str, Any]) -> bool:
        if not slot:
            return False
        if "clinic_id" not in slot:
            return False
        if "time" not in slot:
            return False
        return True
