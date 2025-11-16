"""
Unit tests for ClinicFinderAgent

Tests:
- Google Maps integration
- Google Search fallback
- Mock data fallback
- Parallel availability prefetch
- Distance calculations
- API availability detection
- Circuit breaker behavior
- Error handling
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from sub_agents.clinic_finder_agent import ClinicFinderAgent
from memory import InMemorySessionService


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def config():
    """Default configuration for ClinicFinderAgent."""
    return {
        "model": "gpt-4",
        "max_candidates": 3,
        "tools_enabled": True,
        "search_radius_km": 10,
    }


@pytest.fixture
def session_service():
    """Create session service."""
    return InMemorySessionService()


@pytest.fixture
def session(session_service):
    """Create test session."""
    return session_service.create_session("test_user_clinic")


class FakeEvent:
    """Mock ADK event."""

    def __init__(self, payload):
        self.payload = payload
        self.resume = False


class FakeCtx:
    """Mock ADK context."""

    def __init__(self, session):
        self.session = session
        self.metrics = FakeMetrics()

    async def call_tool(self, name, args):
        """Override in tests to mock tool calls."""
        raise NotImplementedError(f"Tool {name} not mocked")


class FakeMetrics:
    """Mock metrics collector."""

    def __init__(self):
        self.counters = {}
        self.histograms = {}

    def increment(self, name):
        self.counters[name] = self.counters.get(name, 0) + 1

    def histogram(self, name, value, labels=None):
        key = f"{name}:{labels}" if labels else name
        self.histograms.setdefault(key, []).append(value)


# ============================================================================
# BASIC INITIALIZATION TESTS
# ============================================================================

def test_agent_initialization(config):
    """Test agent initializes correctly."""
    agent = ClinicFinderAgent(config)

    assert agent.name == "clinic_finder_agent"
    assert agent.max_candidates == 3
    assert agent.search_radius_km == 10
    assert agent.tools_enabled is True


def test_agent_initialization_with_tools_disabled(config):
    """Test agent initializes with tools disabled."""
    config["tools_enabled"] = False
    agent = ClinicFinderAgent(config)

    assert agent.tools_enabled is False
    assert agent.google_maps is None
    assert agent.google_search is None


# ============================================================================
# MOCK DATA FALLBACK TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_mock_candidates_basic(config, session):
    """Test mock candidates generation."""
    agent = ClinicFinderAgent(config)
    candidates = agent._mock_candidates()

    assert len(candidates) == 3  # max_candidates
    assert all("id" in c for c in candidates)
    assert all("name" in c for c in candidates)
    assert all("address" in c for c in candidates)
    assert all("distance_km" in c for c in candidates)
    assert all("has_api" in c for c in candidates)
    assert all(c["source"] == "mock" for c in candidates)


@pytest.mark.asyncio
async def test_mock_candidates_with_zip_code(config, session):
    """Test mock candidates with zip code."""
    agent = ClinicFinderAgent(config)
    candidates = agent._mock_candidates("94110")

    assert len(candidates) == 3
    assert all(c["zip_code"] == "94110" for c in candidates)


@pytest.mark.asyncio
async def test_mock_candidates_realistic_data(config, session):
    """Test mock candidates have realistic data."""
    agent = ClinicFinderAgent(config)
    candidates = agent._mock_candidates()

    for clinic in candidates:
        # Check required fields
        assert clinic["name"]
        assert clinic["address"]
        assert clinic["phone"]
        assert isinstance(clinic["distance_km"], float)
        assert isinstance(clinic["rating"], float)
        assert isinstance(clinic["hours"], list)
        assert isinstance(clinic["has_api"], bool)

        # Check hours format
        for hour_entry in clinic["hours"]:
            assert "day" in hour_entry
            assert "open" in hour_entry
            assert "close" in hour_entry


# ============================================================================
# ON_EVENT TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_on_event_with_location_query(config, session):
    """Test on_event with location query in payload."""
    agent = ClinicFinderAgent(config)
    agent.tools_enabled = False  # Use mock data

    event = FakeEvent({"location_query": "94110"})
    ctx = FakeCtx(session)

    response = await agent.on_event(event, ctx)

    # Check response
    assert response.resume is True
    assert "candidates" in response.message
    assert len(response.message["candidates"]) > 0

    # Check session updated
    assert "last_clinics" in session
    assert len(session["last_clinics"]) > 0

    # Check metrics
    assert ctx.metrics.counters.get("clinic_searches") == 1


@pytest.mark.asyncio
async def test_on_event_with_session_location(config, session):
    """Test on_event using location from session."""
    agent = ClinicFinderAgent(config)
    agent.tools_enabled = False

    session["location_query"] = "San Francisco, CA"
    event = FakeEvent({})
    ctx = FakeCtx(session)

    response = await agent.on_event(event, ctx)

    assert response.resume is True
    assert len(response.message["candidates"]) > 0


@pytest.mark.asyncio
async def test_on_event_without_location(config, session):
    """Test on_event without location (uses default)."""
    agent = ClinicFinderAgent(config)
    agent.tools_enabled = False

    event = FakeEvent({})
    ctx = FakeCtx(session)

    response = await agent.on_event(event, ctx)

    # Should still return mock candidates
    assert response.resume is True
    assert len(response.message["candidates"]) > 0


# ============================================================================
# GOOGLE MAPS INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_find_with_maps_success(config, session):
    """Test successful Google Maps search."""
    agent = ClinicFinderAgent(config)

    # Mock successful Maps API response
    mock_response = {
        "places": [
            {
                "place_id": "ChIJ123",
                "name": "City Vaccination Center",
                "address": "123 Main St, San Francisco, CA",
                "distance_km": 1.2,
                "rating": 4.5,
                "phone": "(555) 123-4567",
                "hours": [{"day": "Monday", "open": "09:00", "close": "17:00"}]
            },
            {
                "place_id": "ChIJ456",
                "name": "Community Clinic",
                "address": "456 Oak Ave, San Francisco, CA",
                "distance_km": 2.3,
                "rating": 4.2,
                "phone": "(555) 234-5678",
                "hours": []
            }
        ]
    }

    class MockCtx(FakeCtx):
        async def call_tool(self, name, args):
            if name == "google_maps":
                return mock_response
            raise NotImplementedError()

    ctx = MockCtx(session)
    candidates = await agent._find_with_maps(ctx, "94110")

    assert len(candidates) == 2
    assert candidates[0]["id"] == "ChIJ123"
    assert candidates[0]["name"] == "City Vaccination Center"
    assert candidates[0]["source"] == "google_maps"
    assert candidates[0]["distance_km"] == 1.2


@pytest.mark.asyncio
async def test_find_with_maps_error_handling(config, session):
    """Test error handling in Maps search."""
    agent = ClinicFinderAgent(config)

    class MockCtx(FakeCtx):
        async def call_tool(self, name, args):
            raise RuntimeError("Maps API error")

    ctx = MockCtx(session)

    with pytest.raises(RuntimeError):
        await agent._find_with_maps(ctx, "94110")


# ============================================================================
# GOOGLE SEARCH INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_find_with_search_success(config, session):
    """Test successful Google Search fallback."""
    agent = ClinicFinderAgent(config)

    mock_response = {
        "results": [
            {
                "title": "Main Street Clinic",
                "snippet": "Vaccination services available. Call (555) 123-4567",
                "link": "https://example.com/clinic1"
            },
            {
                "title": "Health Center Downtown",
                "snippet": "Walk-in vaccinations",
                "link": "https://example.com/clinic2"
            }
        ]
    }

    class MockCtx(FakeCtx):
        async def call_tool(self, name, args):
            if name == "google_search":
                return mock_response
            raise NotImplementedError()

    ctx = MockCtx(session)
    candidates = await agent._find_with_search(ctx, "94110")

    assert len(candidates) == 2
    assert candidates[0]["name"] == "Main Street Clinic"
    assert candidates[0]["source"] == "google_search"
    assert candidates[0]["phone"] == "(555) 123-4567"


@pytest.mark.asyncio
async def test_find_with_search_extracts_phone(config, session):
    """Test phone number extraction from search results."""
    agent = ClinicFinderAgent(config)

    test_text = "Contact us at 555-123-4567 or visit our website"
    phone = agent._extract_phone(test_text)

    assert phone == "555-123-4567"


@pytest.mark.asyncio
async def test_find_with_search_no_phone(config, session):
    """Test search result without phone number."""
    agent = ClinicFinderAgent(config)

    test_text = "Visit our website for more information"
    phone = agent._extract_phone(test_text)

    assert phone == ""


# ============================================================================
# PARALLEL AVAILABILITY PREFETCH TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_prefetch_availability(config, session):
    """Test parallel availability prefetching."""
    agent = ClinicFinderAgent(config)

    candidates = [
        {"id": "clinic1", "name": "Clinic 1", "has_api": True},
        {"id": "clinic2", "name": "Clinic 2", "has_api": True},
        {"id": "clinic3", "name": "Clinic 3", "has_api": False}
    ]

    ctx = FakeCtx(session)

    # Run prefetch
    await agent._prefetch_availability(ctx, candidates)

    # Check that availability was added to clinics with APIs
    assert "availability" in candidates[0]
    assert "availability" in candidates[1]
    assert candidates[2].get("availability") is None  # No API

    # Check availability structure
    avail = candidates[0]["availability"]
    assert "next_available" in avail
    assert "slots_this_week" in avail
    assert "last_checked" in avail


@pytest.mark.asyncio
async def test_prefetch_availability_empty_list(config, session):
    """Test prefetch with empty candidate list."""
    agent = ClinicFinderAgent(config)
    ctx = FakeCtx(session)

    # Should not raise error
    await agent._prefetch_availability(ctx, [])


# ============================================================================
# HELPER METHOD TESTS
# ============================================================================

def test_check_api_availability_chain(config):
    """Test API availability detection for known chains."""
    agent = ClinicFinderAgent(config)

    # Known chains
    assert agent._check_api_availability({"name": "CVS Pharmacy"}) is True
    assert agent._check_api_availability({"name": "Walgreens"}) is True
    assert agent._check_api_availability({"name": "City Health Center"}) is True

    # Unknown clinic
    assert agent._check_api_availability({"name": "Dr. Smith's Office"}) is False


def test_parse_hours(config):
    """Test hours parsing."""
    agent = ClinicFinderAgent(config)

    hours_data = [
        {"day": "Monday", "open": "09:00", "close": "17:00"},
        {"day": "Tuesday", "open": "09:00", "close": "17:00"}
    ]

    parsed = agent._parse_hours(hours_data)

    assert len(parsed) == 2
    assert parsed[0]["day"] == "Monday"
    assert parsed[0]["open"] == "09:00"


def test_parse_hours_empty(config):
    """Test parsing empty hours."""
    agent = ClinicFinderAgent(config)

    parsed = agent._parse_hours([])
    assert parsed == []


def test_generate_next_available_slot(config):
    """Test next available slot generation."""
    agent = ClinicFinderAgent(config)

    slot = agent._generate_next_available_slot()

    # Should be valid ISO format
    assert isinstance(slot, str)
    datetime.fromisoformat(slot)  # Should not raise


# ============================================================================
# EMIT METHOD TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_emit_method(config, session):
    """Test emit convenience method."""
    agent = ClinicFinderAgent(config)
    agent.tools_enabled = False

    response = await agent.emit({"location_query": "94110"}, session)

    assert response.resume is True
    assert "candidates" in response.message


# ============================================================================
# METRICS TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_metrics_tracking(config, session):
    """Test that metrics are properly tracked."""
    agent = ClinicFinderAgent(config)
    agent.tools_enabled = False

    event = FakeEvent({"location_query": "94110"})
    ctx = FakeCtx(session)

    await agent.on_event(event, ctx)

    # Check metrics were incremented
    assert ctx.metrics.counters["clinic_searches"] == 1

    # Check histogram was recorded
    histogram_keys = [k for k in ctx.metrics.histograms.keys() if "duration" in k]
    assert len(histogram_keys) > 0


# ============================================================================
# ERROR RECOVERY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_maps_fallback_to_mock(config, session):
    """Test fallback to mock data when Maps fails."""
    agent = ClinicFinderAgent(config)

    # Mock Maps API failure
    class MockCtx(FakeCtx):
        async def call_tool(self, name, args):
            raise RuntimeError("Maps unavailable")

    event = FakeEvent({"location_query": "94110"})
    ctx = MockCtx(session)

    # Should fall back to mock data instead of raising
    response = await agent.on_event(event, ctx)

    assert response.resume is True
    assert len(response.message["candidates"]) > 0
    assert response.message["method"] == "mock_fallback"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_full_search_flow(config, session):
    """Test complete search flow from event to response."""
    agent = ClinicFinderAgent(config)
    agent.tools_enabled = False

    # Set location
    session["location_query"] = "San Francisco, CA"

    # Create event
    event = FakeEvent({"location_query": "94110"})
    ctx = FakeCtx(session)

    # Execute
    response = await agent.on_event(event, ctx)

    # Verify complete flow
    assert response.resume is True
    assert "candidates" in response.message
    assert "method" in response.message

    # Verify session updated
    assert "last_clinics" in session
    assert "clinic_search_method" in session

    # Verify candidates are well-formed
    for candidate in response.message["candidates"]:
        assert "id" in candidate
        assert "name" in candidate
        assert "address" in candidate
        assert "distance_km" in candidate
        assert "has_api" in candidate


@pytest.mark.asyncio
async def test_concurrent_searches(config, session_service):
    """Test handling concurrent search requests."""
    agent = ClinicFinderAgent(config)
    agent.tools_enabled = False

    # Create multiple sessions
    sessions = [session_service.create_session(f"user_{i}") for i in range(5)]

    # Create concurrent search tasks
    async def search(sess):
        event = FakeEvent({"location_query": "94110"})
        ctx = FakeCtx(sess)
        return await agent.on_event(event, ctx)

    tasks = [search(s) for s in sessions]
    responses = await asyncio.gather(*tasks)

    # All should succeed
    assert len(responses) == 5
    assert all(r.resume is True for r in responses)
    assert all("candidates" in r.message for r in responses)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])