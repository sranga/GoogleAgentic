"""
Pytest Configuration

Handles Python path configuration and provides common fixtures for tests.
"""

import sys
import os
import pytest
import asyncio

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


@pytest.fixture
def mock_event():
    """Factory for MockEvent."""
    return lambda payload=None, resume=False: MockEvent(payload, resume)


@pytest.fixture
def mock_ctx(session):
    """MockCtx with test session."""
    return MockCtx(session)


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "security: marks security tests")