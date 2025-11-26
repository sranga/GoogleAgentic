"""
Pytest Configuration

Handles Python path configuration and provides common fixtures for tests.
Includes mock agents for testing orchestrator without real agent dependencies.
"""

import sys
import os
import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

# Add src directory to Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def config():
    """Default test configuration."""
    return {
        "model": "gemini-2.0-flash",
        "followup_seconds": 5,
        "max_candidates": 3,
        "max_retries": 3,
        "max_history_length": 8,
        "search_radius_km": 10,
    }


@pytest.fixture
def memory_bank():
    """Fresh MemoryBank instance."""
    from memory import MemoryBank
    return MemoryBank()


@pytest.fixture
def session_service():
    """Fresh InMemorySessionService instance."""
    from memory import InMemorySessionService
    return InMemorySessionService()


@pytest.fixture
def session(session_service):
    """Test session."""
    return session_service.create_session("test_user")


# ============================================================================
# MOCK EVENT AND CONTEXT CLASSES
# ============================================================================

class MockEvent:
    """Mock ADK event for testing."""
    def __init__(self, payload=None, resume=False):
        self.payload = payload or {}
        self.resume = resume


class MockMetrics:
    """Mock metrics collector."""
    def __init__(self):
        self.counters = {}
        self.histograms = {}

    def increment(self, name, labels=None):
        key = f"{name}:{labels}" if labels else name
        self.counters[key] = self.counters.get(key, 0) + 1

    def histogram(self, name, value, labels=None):
        key = f"{name}:{labels}" if labels else name
        self.histograms.setdefault(key, []).append(value)

    def counter(self, name, value=1, labels=None):
        self.increment(name, labels)


class MockCtx:
    """Mock ADK context for testing."""
    def __init__(self, session):
        self.session = session
        self.metrics = MockMetrics()
        self.tool_results = {}
        self.model_response = "Mock LLM response"

    async def call_tool(self, name, args):
        if name in self.tool_results:
            return self.tool_results[name]
        raise NotImplementedError(f"Tool {name} not mocked")

    async def call_model(self, prompt):
        return self.model_response


# ============================================================================
# MOCK AGENT CLASSES - For testing orchestrator
# ============================================================================

class MockVaccineInfoAgent:
    """Mock VaccineInfoAgent for testing."""

    def __init__(self, config=None, memory_bank=None):
        self.config = config or {}
        self.memory_bank = memory_bank
        self.response_text = "Mock vaccine information response"
        self.should_raise_error = False
        self.error_message = "Mock error"

    async def emit(self, payload: Dict[str, Any], session: Dict[str, Any]):
        """Mock emit method."""
        if self.should_raise_error:
            raise Exception(self.error_message)

        return type("Response", (), {
            "text": self.response_text,
            "message": {"text": self.response_text}
        })()


class MockClinicFinderAgent:
    """Mock ClinicFinderAgent for testing."""

    def __init__(self, config=None):
        self.config = config or {}
        self.candidates = [
            {
                "id": "clinic_1",
                "name": "Test Clinic 1",
                "address": "123 Main St",
                "has_api": True,
                "distance_km": 1.5,
                "rating": 4.5
            }
        ]
        self.should_raise_error = False
        self.error_message = "Mock search error"

    async def emit(self, payload: Dict[str, Any], session: Dict[str, Any]):
        """Mock emit method."""
        if self.should_raise_error:
            raise Exception(self.error_message)

        return {
            "candidates": self.candidates,
            "method": "mock"
        }


class MockAppointmentAgent:
    """Mock AppointmentAgent for testing."""

    def __init__(self, config=None):
        self.config = config or {}
        self.confirmation = {
            "confirmed": True,
            "confirmation_id": f"MOCK-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "clinic_id": "clinic_1",
            "slot_time": datetime.utcnow().isoformat()
        }
        self.should_raise_error = False
        self.error_message = "Mock booking error"

    async def emit(self, payload: Dict[str, Any], session: Dict[str, Any]):
        """Mock emit method."""
        if self.should_raise_error:
            raise Exception(self.error_message)

        return self.confirmation


class MockFollowUpAgent:
    """Mock FollowUpAgent for testing."""

    def __init__(self, config=None, memory_bank=None):
        self.config = config or {}
        self.memory_bank = memory_bank
        self.should_raise_error = False
        self.error_message = "Mock followup error"

    async def emit(self, payload: Dict[str, Any], session: Dict[str, Any]):
        """Mock emit method."""
        if self.should_raise_error:
            raise Exception(self.error_message)

        return {"scheduled": True}


class MockAnalyticsAgent:
    """Mock AnalyticsAgent for testing."""

    def __init__(self, config=None, memory_bank=None):
        self.config = config or {}
        self.memory_bank = memory_bank
        self.records = []

    async def emit(self, payload: Dict[str, Any], session: Optional[Dict[str, Any]]):
        """Mock emit method."""
        if payload.get("action") == "ingest":
            self.records.append(payload.get("record"))
        return {"success": True}


# ============================================================================
# AGENT MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_vaccine_info():
    """Mock VaccineInfoAgent."""
    return MockVaccineInfoAgent()


@pytest.fixture
def mock_clinic_finder():
    """Mock ClinicFinderAgent."""
    return MockClinicFinderAgent()


@pytest.fixture
def mock_appointment_agent():
    """Mock AppointmentAgent."""
    return MockAppointmentAgent()


@pytest.fixture
def mock_followup_agent():
    """Mock FollowUpAgent."""
    return MockFollowUpAgent()


@pytest.fixture
def mock_analytics_agent():
    """Mock AnalyticsAgent."""
    return MockAnalyticsAgent()


@pytest.fixture
def all_mock_agents(mock_vaccine_info, mock_clinic_finder, mock_appointment_agent,
                    mock_followup_agent, mock_analytics_agent):
    """Dictionary of all mock agents."""
    return {
        "vaccine_info": mock_vaccine_info,
        "clinic_finder": mock_clinic_finder,
        "appointment_agent": mock_appointment_agent,
        "followup_agent": mock_followup_agent,
        "analytics_agent": mock_analytics_agent
    }


# ============================================================================
# HELPER FIXTURES
# ============================================================================

@pytest.fixture
def mock_event():
    """Factory for MockEvent."""
    return lambda payload=None, resume=False: MockEvent(payload, resume)


@pytest.fixture
def mock_ctx(session):
    """MockCtx with test session."""
    return MockCtx(session)


@pytest.fixture
def replace_orchestrator_agents(all_mock_agents):
    """
    Context manager to temporarily replace orchestrator agents with mocks.

    Usage:
        with replace_orchestrator_agents(orchestrator, all_mock_agents):
            # orchestrator now uses mock agents
            result = await orchestrator.find_and_schedule(session, "94110")
        # original agents restored
    """
    from contextlib import contextmanager

    @contextmanager
    def replacer(orchestrator, agents=None):
        agents = agents or all_mock_agents

        # Save originals
        originals = {
            "vaccine_info": orchestrator.vaccine_info,
            "clinic_finder": orchestrator.clinic_finder,
            "appointment_agent": orchestrator.appointment_agent,
            "followup_agent": orchestrator.followup_agent,
            "analytics_agent": orchestrator.analytics_agent
        }

        # Replace with mocks
        orchestrator.vaccine_info = agents["vaccine_info"]
        orchestrator.clinic_finder = agents["clinic_finder"]
        orchestrator.appointment_agent = agents["appointment_agent"]
        orchestrator.followup_agent = agents["followup_agent"]
        orchestrator.analytics_agent = agents["analytics_agent"]

        try:
            yield orchestrator
        finally:
            # Restore originals
            orchestrator.vaccine_info = originals["vaccine_info"]
            orchestrator.clinic_finder = originals["clinic_finder"]
            orchestrator.appointment_agent = originals["appointment_agent"]
            orchestrator.followup_agent = originals["followup_agent"]
            orchestrator.analytics_agent = originals["analytics_agent"]

    return replacer


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "security: marks security tests")