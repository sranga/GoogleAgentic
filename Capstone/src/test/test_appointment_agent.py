"""
Tests for AppointmentAgent
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sub_agents.appointment_agent import AppointmentAgent
from memory import InMemorySessionService


@pytest.fixture
def agent(config):
    return AppointmentAgent(config)


@pytest.fixture
def session():
    sess = InMemorySessionService().create_session("test_user")
    sess["last_clinics"] = [
        {"id": "clinic_1", "name": "Test Clinic 1", "has_api": True},
        {"id": "clinic_2", "name": "Test Clinic 2", "has_api": True},
        {"id": "clinic_3", "name": "Test Clinic 3", "has_api": False},
    ]
    return sess


class TestAppointmentAgentInit:
    def test_initialization(self, config):
        agent = AppointmentAgent(config)
        assert agent.name == "appointment_agent"
        assert agent.max_retries == config["max_retries"]

    def test_has_booking_tool(self, agent):
        assert agent.booking_tool is not None
        tool_names = [t.name for t in agent.tools]
        assert "booking_api" in tool_names


class TestSlotValidation:
    def test_valid_slot(self, agent):
        slot = {"clinic_id": "c1", "time": "2025-12-01T10:00:00Z"}
        assert agent._validate_slot(slot) is True

    def test_missing_clinic_id(self, agent):
        slot = {"time": "2025-12-01T10:00:00Z"}
        assert agent._validate_slot(slot) is False

    def test_missing_time(self, agent):
        slot = {"clinic_id": "c1"}
        assert agent._validate_slot(slot) is False

    def test_none_slot(self, agent):
        assert agent._validate_slot(None) is False

    def test_empty_slot(self, agent):
        assert agent._validate_slot({}) is False


class TestSlotSelection:
    def test_choose_slot_prefers_api_clinics(self, agent, session):
        slot = agent._choose_slot(session["last_clinics"])
        assert slot is not None
        assert slot["clinic_id"] in ["clinic_1", "clinic_2"]

    def test_choose_slot_empty_list(self, agent):
        slot = agent._choose_slot([])
        assert slot is None

    def test_choose_slot_no_api_clinics(self, agent):
        clinics = [{"id": "c1", "has_api": False}]
        slot = agent._choose_slot(clinics)
        assert slot is None

    def test_choose_slot_includes_time(self, agent, session):
        slot = agent._choose_slot(session["last_clinics"])
        assert "time" in slot


class TestGenerateDefaultSlot:
    def test_generates_valid_iso_time(self, agent):
        slot_time = agent._generate_default_slot()
        from datetime import datetime
        # Should parse without error
        datetime.fromisoformat(slot_time.replace("Z", "+00:00"))


class TestOnEvent:
    @pytest.mark.asyncio
    async def test_successful_booking(self, agent, session):
        event = MockEvent({})
        ctx = MockCtx(session)
        ctx.tool_results["booking_api"] = {
            "confirmed": True,
            "confirmation_id": "CONF-123",
            "clinic_id": "clinic_1",
            "time": "2025-12-01T10:00:00Z"
        }

        response = await agent.on_event(event, ctx)

        assert response["confirmed"] is True
        assert response["confirmation_id"] == "CONF-123"

    @pytest.mark.asyncio
    async def test_resets_retry_counter_on_success(self, agent, session):
        session["appointment_attempt"] = 2
        event = MockEvent({})
        ctx = MockCtx(session)
        ctx.tool_results["booking_api"] = {"confirmed": True, "confirmation_id": "C1"}

        await agent.on_event(event, ctx)

        assert session["appointment_attempt"] == 1

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, agent, session):
        session["appointment_attempt"] = 4  # max_retries + 1
        session["last_clinics"] = []

        event = MockEvent({})
        ctx = MockCtx(session)

        response = await agent.on_event(event, ctx)

        assert hasattr(response, 'text')
        assert "wasn't able to confirm" in response.text.lower()

    @pytest.mark.asyncio
    async def test_retry_on_invalid_slot(self, agent, session):
        session["last_clinics"] = []  # Force invalid slot

        event = MockEvent({})
        ctx = MockCtx(session)

        response = await agent.on_event(event, ctx)

        assert response.restart is True
        assert session["appointment_attempt"] == 2

    @pytest.mark.asyncio
    async def test_tracks_metrics_on_success(self, agent, session):
        event = MockEvent({})
        ctx = MockCtx(session)
        ctx.tool_results["booking_api"] = {"confirmed": True, "confirmation_id": "C1"}

        await agent.on_event(event, ctx)

        assert "appointments_booked:None" in ctx.metrics.counters


class TestToolInvocation:
    @pytest.mark.asyncio
    async def test_tool_called_with_correct_args(self, agent, session):
        captured_args = {}

        class CapturingCtx(MockCtx):
            async def call_tool(self, name, args):
                captured_args[name] = args
                return {"confirmed": True, "confirmation_id": "C1"}

        event = MockEvent({})
        ctx = CapturingCtx(session)

        await agent.on_event(event, ctx)

        assert "booking_api" in captured_args
        assert "clinic_id" in captured_args["booking_api"]
        assert "time" in captured_args["booking_api"]
        assert "user_id" in captured_args["booking_api"]

    @pytest.mark.asyncio
    async def test_tool_failure_triggers_retry(self, agent, session):
        class FailingCtx(MockCtx):
            async def call_tool(self, name, args):
                raise RuntimeError("Booking API error")

        event = MockEvent({})
        ctx = FailingCtx(session)

        response = await agent.on_event(event, ctx)

        assert response.restart is True
        assert session["appointment_attempt"] == 2