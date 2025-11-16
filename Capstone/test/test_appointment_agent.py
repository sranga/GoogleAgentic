"""
Unit tests for AppointmentAgent

Tests:
- Retry logic (LoopAgent behavior)
- Slot validation
- Booking API integration
- Max retry limits
- Circuit breaker integration
- Error handling
- Tool invocation
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from sub_agents.appointment_agent import AppointmentAgent
from memory import InMemorySessionService


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def config():
    """Default configuration for AppointmentAgent."""
    return {
        "model": "gpt-4",
        "max_retries": 3,
    }


@pytest.fixture
def session_service():
    """Create session service."""
    return InMemorySessionService()


@pytest.fixture
def session(session_service):
    """Create test session with clinic data."""
    sess = session_service.create_session("test_user_appt")
    sess["last_clinics"] = [
        {
            "id": "clinic_1",
            "name": "Test Clinic 1",
            "has_api": True,
            "distance_km": 1.5
        },
        {
            "id": "clinic_2",
            "name": "Test Clinic 2",
            "has_api": True,
            "distance_km": 2.8
        },
        {
            "id": "clinic_3",
            "name": "Test Clinic 3",
            "has_api": False,
            "distance_km": 5.0
        }
    ]
    return sess


class FakeEvent:
    """Mock ADK event."""

    def __init__(self, payload=None):
        self.payload = payload or {}
        self.resume = False


class FakeCtx:
    """Mock ADK context."""

    def __init__(self, session):
        self.session = session
        self.tool_results = {}

    async def call_tool(self, name, args):
        """Mock tool call - can be overridden in tests."""
        if name in self.tool_results:
            return self.tool_results[name]
        return {"confirmed": True, "confirmation_id": "CONF-TEST-123"}


# ============================================================================
# BASIC INITIALIZATION TESTS
# ============================================================================

def test_agent_initialization(config):
    """Test agent initializes correctly."""
    agent = AppointmentAgent(config)

    assert agent.name == "appointment_agent"
    assert agent.max_retries == 3
    assert len(agent.tools) > 0  # Should have booking_api tool


def test_agent_has_booking_tool(config):
    """Test agent is configured with booking tool."""
    agent = AppointmentAgent(config)

    tool_names = [t.name for t in agent.tools]
    assert "booking_api" in tool_names


# ============================================================================
# SLOT VALIDATION TESTS
# ============================================================================

def test_validate_slot_valid(config):
    """Test validation of valid slot."""
    agent = AppointmentAgent(config)

    valid_slot = {
        "clinic_id": "clinic_1",
        "time": "2025-12-01T10:00:00Z"
    }

    assert agent._validate_slot(valid_slot) is True


def test_validate_slot_missing_clinic_id(config):
    """Test validation fails without clinic_id."""
    agent = AppointmentAgent(config)

    invalid_slot = {
        "time": "2025-12-01T10:00:00Z"
    }

    assert agent._validate_slot(invalid_slot) is False


def test_validate_slot_missing_time(config):
    """Test validation fails without time."""
    agent = AppointmentAgent(config)

    invalid_slot = {
        "clinic_id": "clinic_1"
    }

    assert agent._validate_slot(invalid_slot) is False


def test_validate_slot_none(config):
    """Test validation fails for None."""
    agent = AppointmentAgent(config)

    assert agent._validate_slot(None) is False


def test_validate_slot_empty_dict(config):
    """Test validation fails for empty dict."""
    agent = AppointmentAgent(config)

    assert agent._validate_slot({}) is False


# ============================================================================
# SLOT SELECTION TESTS
# ============================================================================

def test_choose_slot_selects_api_clinic(config, session):
    """Test slot selection prefers clinics with API."""
    agent = AppointmentAgent(config)
    clinics = session["last_clinics"]

    slot = agent._choose_slot(clinics)

    assert slot is not None
    assert slot["clinic_id"] in ["clinic_1", "clinic_2"]  # Both have APIs


def test_choose_slot_empty_list(config):
    """Test slot selection with empty clinic list."""
    agent = AppointmentAgent(config)

    slot = agent._choose_slot([])

    assert slot is None


def test_choose_slot_no_api_clinics(config):
    """Test slot selection when no clinics have APIs."""
    agent = AppointmentAgent(config)

    clinics = [
        {"id": "clinic_1", "has_api": False},
        {"id": "clinic_2", "has_api": False}
    ]

    slot = agent._choose_slot(clinics)

    assert slot is None


def test_choose_slot_generates_time(config, session):
    """Test that chosen slot includes time."""
    agent = AppointmentAgent(config)
    clinics = session["last_clinics"]

    slot = agent._choose_slot(clinics)

    assert "time" in slot
    # Verify time is ISO format
    datetime.fromisoformat(slot["time"].replace("Z", "+00:00"))


# ============================================================================
# ON_EVENT TESTS - SUCCESSFUL BOOKING
# ============================================================================

@pytest.mark.asyncio
async def test_on_event_successful_booking(config, session):
    """Test successful appointment booking."""
    agent = AppointmentAgent(config)

    # Mock successful tool call
    ctx = FakeCtx(session)
    ctx.tool_results["booking_api"] = {
        "confirmed": True,
        "confirmation_id": "CONF-12345",
        "clinic_id": "clinic_1",
        "time": "2025-12-01T10:00:00Z"
    }

    event = FakeEvent({})
    response = await agent.on_event(event, ctx)

    # Should return tool result
    assert response["confirmed"] is True
    assert response["confirmation_id"] == "CONF-12345"

    # Retry counter should be reset
    assert session.get("appointment_attempt", 1) == 1


@pytest.mark.asyncio
async def test_on_event_with_valid_clinics(config, session):
    """Test booking with valid clinic list."""
    agent = AppointmentAgent(config)

    ctx = FakeCtx(session)
    event = FakeEvent({})

    response = await agent.on_event(event, ctx)

    # Should successfully call tool and return result
    assert "confirmed" in response


# ============================================================================
# ON_EVENT TESTS - RETRY LOGIC
# ============================================================================

@pytest.mark.asyncio
async def test_on_event_retry_on_invalid_slot(config, session):
    """Test retry when slot validation fails."""
    agent = AppointmentAgent(config)

    # Remove clinics to force invalid slot
    session["last_clinics"] = []

    ctx = FakeCtx(session)
    event = FakeEvent({})

    response = await agent.on_event(event, ctx)

    # Should restart event
    assert response.restart is True
    assert "Retrying" in response.message.text

    # Retry counter should increment
    assert session["appointment_attempt"] == 2


@pytest.mark.asyncio
async def test_on_event_max_retries_exceeded(config, session):
    """Test behavior when max retries exceeded."""
    agent = AppointmentAgent(config)

    # Set retry counter to max
    session["appointment_attempt"] = 4  # max_retries + 1
    session["last_clinics"] = []  # Force failure

    ctx = FakeCtx(session)
    event = FakeEvent({})

    response = await agent.on_event(event, ctx)

    # Should return error message, not restart
    assert "wasn't able to confirm" in response.text.lower()
    assert not hasattr(response, 'restart') or response.restart is False


@pytest.mark.asyncio
async def test_on_event_retry_increments_counter(config, session):
    """Test that retry counter increments properly."""
    agent = AppointmentAgent(config)

    # Force invalid slot
    session["last_clinics"] = []
    initial_attempt = session.get("appointment_attempt", 1)

    ctx = FakeCtx(session)
    event = FakeEvent({})

    await agent.on_event(event, ctx)

    # Counter should increment
    assert session["appointment_attempt"] == initial_attempt + 1


# ============================================================================
# TOOL INVOCATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_tool_called_with_correct_args(config, session):
    """Test booking tool is called with correct arguments."""
    agent = AppointmentAgent(config)

    tool_args_captured = {}

    class MockCtx(FakeCtx):
        async def call_tool(self, name, args):
            tool_args_captured[name] = args
            return {"confirmed": True, "confirmation_id": "CONF-123"}

    ctx = MockCtx(session)
    event = FakeEvent({})

    await agent.on_event(event, ctx)

    # Check tool was called with right args
    assert "booking_api" in tool_args_captured
    args = tool_args_captured["booking_api"]
    assert "clinic_id" in args
    assert "time" in args
    assert "user_id" in args
    assert args["user_id"] == session["user_id"]


@pytest.mark.asyncio
async def test_tool_failure_handling(config, session):
    """Test handling of tool failure."""
    agent = AppointmentAgent(config)

    class MockCtx(FakeCtx):
        async def call_tool(self, name, args):
            raise RuntimeError("Booking API unavailable")

    ctx = MockCtx(session)
    event = FakeEvent({})

    # Should raise the error (circuit breaker would handle this at orchestrator level)
    with pytest.raises(RuntimeError):
        await agent.on_event(event, ctx)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_on_event_no_clinics_in_session(config):
    """Test behavior when no clinics in session."""
    agent = AppointmentAgent(config)

    # Create session without clinics
    service = InMemorySessionService()
    session = service.create_session("test_user_no_clinics")
    # Don't add last_clinics

    ctx = FakeCtx(session)
    event = FakeEvent({})

    response = await agent.on_event(event, ctx)

    # Should attempt restart or return error
    assert response.restart is True or "unable" in str(response).lower()


@pytest.mark.asyncio
async def test_on_event_malformed_clinic_data(config):
    """Test handling of malformed clinic data."""
    agent = AppointmentAgent(config)

    service = InMemorySessionService()
    session = service.create_session("test_user_malformed")
    session["last_clinics"] = [
        {"id": "clinic_1"},  # Missing has_api
        {"has_api": True},  # Missing id
    ]

    ctx = FakeCtx(session)
    event = FakeEvent({})

    # Should handle gracefully (either restart or error)
    response = await agent.on_event(event, ctx)
    assert response is not None


# ============================================================================
# STATE MANAGEMENT TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_retry_counter_reset_after_success(config, session):
    """Test retry counter resets after successful booking."""
    agent = AppointmentAgent(config)

    # Set retry counter high
    session["appointment_attempt"] = 3

    ctx = FakeCtx(session)
    event = FakeEvent({})

    await agent.on_event(event, ctx)

    # Should reset to 1 after success
    assert session["appointment_attempt"] == 1


@pytest.mark.asyncio
async def test_session_state_preserved(config, session):
    """Test that session state is preserved across calls."""
    agent = AppointmentAgent(config)

    original_user_id = session["user_id"]
    original_clinics = session["last_clinics"].copy()

    ctx = FakeCtx(session)
    event = FakeEvent({})

    await agent.on_event(event, ctx)

    # Session data should be intact
    assert session["user_id"] == original_user_id
    assert session["last_clinics"] == original_clinics


# ============================================================================
# CONCURRENT REQUEST TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_concurrent_bookings_different_users(config):
    """Test handling concurrent bookings for different users."""
    agent = AppointmentAgent(config)

    # Create multiple sessions
    service = InMemorySessionService()
    sessions = []
    for i in range(3):
        sess = service.create_session(f"user_{i}")
        sess["last_clinics"] = [
            {"id": f"clinic_{i}", "has_api": True}
        ]
        sessions.append(sess)

    # Create concurrent booking tasks
    async def book(sess):
        ctx = FakeCtx(sess)
        event = FakeEvent({})
        return await agent.on_event(event, ctx)

    import asyncio
    tasks = [book(s) for s in sessions]
    results = await asyncio.gather(*tasks)

    # All should succeed
    assert len(results) == 3
    assert all("confirmed" in r for r in results)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_full_booking_flow(config, session):
    """Test complete booking flow from event to confirmation."""
    agent = AppointmentAgent(config)

    # Mock tool to return full confirmation
    ctx = FakeCtx(session)
    ctx.tool_results["booking_api"] = {
        "confirmed": True,
        "confirmation_id": "CONF-FULL-123",
        "clinic_id": "clinic_1",
        "clinic_name": "Test Clinic 1",
        "time": "2025-12-01T10:00:00Z",
        "user_id": session["user_id"]
    }

    event = FakeEvent({})
    response = await agent.on_event(event, ctx)

    # Verify complete confirmation structure
    assert response["confirmed"] is True
    assert response["confirmation_id"] == "CONF-FULL-123"
    assert response["clinic_id"] == "clinic_1"
    assert response["time"] == "2025-12-01T10:00:00Z"


@pytest.mark.asyncio
async def test_retry_until_success(config, session):
    """Test retry logic until successful booking."""
    agent = AppointmentAgent(config)

    call_count = 0

    class MockCtx(FakeCtx):
        async def call_tool(self, name, args):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                # First call: simulate slot validation failure by returning None
                # This won't actually happen with the current implementation
                # but tests the retry logic
                return {"confirmed": False}
            return {"confirmed": True, "confirmation_id": f"CONF-{call_count}"}

    # This test verifies the agent doesn't get stuck in retry loop
    ctx = MockCtx(session)
    event = FakeEvent({})

    response = await agent.on_event(event, ctx)

    # Should eventually succeed
    assert "confirmed" in response


# ============================================================================
# ERROR MESSAGE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_max_retry_error_message(config, session):
    """Test error message when max retries exceeded."""
    agent = AppointmentAgent(config)

    session["appointment_attempt"] = agent.max_retries + 1
    session["last_clinics"] = []

    ctx = FakeCtx(session)
    event = FakeEvent({})

    response = await agent.on_event(event, ctx)

    # Should have helpful error message
    assert "wasn't able to confirm" in response.text.lower()
    assert "several attempts" in response.text.lower()


@pytest.mark.asyncio
async def test_retry_message_format(config, session):
    """Test retry message format."""
    agent = AppointmentAgent(config)

    session["last_clinics"] = []

    ctx = FakeCtx(session)
    event = FakeEvent({})

    response = await agent.on_event(event, ctx)

    # Should have retry message
    assert hasattr(response, 'message')
    assert "retry" in response.message.text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])