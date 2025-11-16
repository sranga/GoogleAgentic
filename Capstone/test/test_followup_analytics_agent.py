"""
Unit tests for FollowUpAgent and AnalyticsAgent

Tests for FollowUpAgent:
- Pause/resume mechanism
- Reminder scheduling
- MemoryBank integration
- Check-in message generation

Tests for AnalyticsAgent:
- Metrics aggregation
- Concurrent record ingestion
- PII sanitization
- Export functionality
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock

from sub_agents.followup_agent import FollowUpAgent
from sub_agents.analytics_agent import AnalyticsAgent, MetricsCollector
from memory import InMemorySessionService, MemoryBank


# ============================================================================
# FOLLOWUP AGENT TESTS
# ============================================================================

@pytest.fixture
def followup_config():
    """Configuration for FollowUpAgent."""
    return {
        "model": "gpt-4",
        "followup_seconds": 5,
    }


@pytest.fixture
def memory_bank():
    """Create MemoryBank instance."""
    return MemoryBank()


@pytest.fixture
def session_service():
    """Create session service."""
    return InMemorySessionService()


@pytest.fixture
def session(session_service):
    """Create test session."""
    return session_service.create_session("test_user_followup")


class FakeEvent:
    """Mock ADK event."""

    def __init__(self, payload=None, resume=False):
        self.payload = payload or {}
        self.resume = resume


class FakeCtx:
    """Mock ADK context."""

    def __init__(self, session):
        self.session = session


# --- FollowUpAgent Basic Tests ---

def test_followup_agent_initialization(followup_config, memory_bank):
    """Test FollowUpAgent initializes correctly."""
    agent = FollowUpAgent(followup_config, memory_bank=memory_bank)

    assert agent.name == "followup_agent"
    assert agent.memory_bank is memory_bank


@pytest.mark.asyncio
async def test_schedule_reminder(followup_config, memory_bank, session):
    """Test scheduling a reminder (initial call)."""
    agent = FollowUpAgent(followup_config, memory_bank=memory_bank)

    # Initial event (not a resume)
    event = FakeEvent(resume=False)
    ctx = FakeCtx(session)

    response = await agent.on_event(event, ctx)

    # Should return EventActions with pause_until
    assert hasattr(response, 'pause_until')
    assert hasattr(response, 'message')
    assert "follow up" in response.message.text.lower()

    # Session should have reminder metadata
    assert "followup_resume_at" in session


@pytest.mark.asyncio
async def test_reminder_timing(followup_config, memory_bank, session):
    """Test reminder is scheduled for correct time."""
    followup_config["followup_seconds"] = 10
    agent = FollowUpAgent(followup_config, memory_bank=memory_bank)

    event = FakeEvent(resume=False)
    ctx = FakeCtx(session)

    before = datetime.utcnow()
    response = await agent.on_event(event, ctx)
    after = datetime.utcnow()

    # Check pause_until is approximately 10 seconds in future
    pause_time = response.pause_until
    assert before + timedelta(seconds=9) <= pause_time <= after + timedelta(seconds=11)


@pytest.mark.asyncio
async def test_handle_checkin(followup_config, memory_bank, session):
    """Test handling check-in (resume event)."""
    agent = FollowUpAgent(followup_config, memory_bank=memory_bank)

    # Resume event
    event = FakeEvent(resume=True)
    ctx = FakeCtx(session)

    response = await agent.on_event(event, ctx)

    # Should return ModelMessage with check-in text
    assert hasattr(response, 'text')
    assert "feeling" in response.text.lower()
    assert "vaccination" in response.text.lower()


@pytest.mark.asyncio
async def test_checkin_stores_in_memory(followup_config, memory_bank, session):
    """Test check-in stores data in MemoryBank."""
    agent = FollowUpAgent(followup_config, memory_bank=memory_bank)

    event = FakeEvent(resume=True)
    ctx = FakeCtx(session)

    await agent.on_event(event, ctx)

    # Check memory bank was updated
    memories = memory_bank.get(session["user_id"])
    assert len(memories) > 0
    assert any("followup_sent_at" in m for m in memories)


@pytest.mark.asyncio
async def test_followup_message_content(followup_config, memory_bank, session):
    """Test follow-up message contains appropriate content."""
    agent = FollowUpAgent(followup_config, memory_bank=memory_bank)

    event = FakeEvent(resume=True)
    ctx = FakeCtx(session)

    response = await agent.on_event(event, ctx)

    text = response.text.lower()
    # Should ask about symptoms
    assert any(word in text for word in ["feeling", "symptoms", "soreness", "fever"])


# ============================================================================
# ANALYTICS AGENT TESTS
# ============================================================================

@pytest.fixture
def analytics_config():
    """Configuration for AnalyticsAgent."""
    return {
        "model": "gpt-4",
    }


# --- MetricsCollector Tests ---

def test_metrics_collector_initialization():
    """Test MetricsCollector initializes correctly."""
    collector = MetricsCollector()

    assert isinstance(collector.counters, dict)
    assert isinstance(collector.gauges, dict)


def test_metrics_collector_increment():
    """Test incrementing counters."""
    collector = MetricsCollector()

    collector.increment("test_counter")
    assert collector.counters["test_counter"] == 1

    collector.increment("test_counter", amount=5)
    assert collector.counters["test_counter"] == 6


def test_metrics_collector_set_gauge():
    """Test setting gauge values."""
    collector = MetricsCollector()

    collector.set_gauge("active_users", 42)
    assert collector.gauges["active_users"] == 42

    collector.set_gauge("active_users", 100)
    assert collector.gauges["active_users"] == 100


def test_metrics_collector_snapshot():
    """Test taking snapshot of metrics."""
    collector = MetricsCollector()

    collector.increment("requests")
    collector.set_gauge("memory_mb", 256)

    snapshot = collector.snapshot()

    assert "counters" in snapshot
    assert "gauges" in snapshot
    assert snapshot["counters"]["requests"] == 1
    assert snapshot["gauges"]["memory_mb"] == 256


def test_metrics_collector_thread_safety():
    """Test MetricsCollector is thread-safe."""
    collector = MetricsCollector()

    def increment_many():
        for _ in range(1000):
            collector.increment("concurrent_test")

    import threading
    threads = [threading.Thread(target=increment_many) for _ in range(5)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should be exactly 5000
    assert collector.counters["concurrent_test"] == 5000


# --- AnalyticsAgent Basic Tests ---

def test_analytics_agent_initialization(analytics_config, memory_bank):
    """Test AnalyticsAgent initializes correctly."""
    agent = AnalyticsAgent(analytics_config, memory_bank=memory_bank)

    assert agent.name == "analytics_agent"
    assert agent.memory_bank is memory_bank
    assert isinstance(agent.metrics, MetricsCollector)


@pytest.mark.asyncio
async def test_ingest_record(analytics_config, memory_bank, session):
    """Test ingesting a single record."""
    agent = AnalyticsAgent(analytics_config, memory_bank=memory_bank)

    record = {
        "event": "appointment_confirmed",
        "clinic_id": "clinic_1",
        "timestamp": datetime.utcnow().isoformat()
    }

    event = FakeEvent(payload={"action": "ingest", "record": record})
    ctx = FakeCtx(session)

    response = await agent.on_event(event, ctx)

    assert response.resume is True
    assert response.message["status"] == "ingested"

    # Check record was stored
    assert len(agent._records) == 1
    assert agent._records[0]["event"] == "appointment_confirmed"


@pytest.mark.asyncio
async def test_ingest_multiple_records(analytics_config, memory_bank, session):
    """Test ingesting multiple records."""
    agent = AnalyticsAgent(analytics_config, memory_bank=memory_bank)
    ctx = FakeCtx(session)

    records = [
        {"event": "appointment_confirmed", "clinic_id": f"clinic_{i}"}
        for i in range(10)
    ]

    for record in records:
        event = FakeEvent(payload={"action": "ingest", "record": record})
        await agent.on_event(event, ctx)

    assert len(agent._records) == 10


@pytest.mark.asyncio
async def test_ingest_without_record(analytics_config, memory_bank, session):
    """Test ingesting without providing record."""
    agent = AnalyticsAgent(analytics_config, memory_bank=memory_bank)

    event = FakeEvent(payload={"action": "ingest"})
    ctx = FakeCtx(session)

    response = await agent.on_event(event, ctx)

    assert "error" in response.message


@pytest.mark.asyncio
async def test_aggregate_records(analytics_config, memory_bank, session):
    """Test aggregating records."""
    agent = AnalyticsAgent(analytics_config, memory_bank=memory_bank)
    ctx = FakeCtx(session)

    # Ingest test data
    records = [
        {"event": "appointment_confirmed", "clinic_id": "clinic_1"},
        {"event": "appointment_confirmed", "clinic_id": "clinic_2"},
        {"event": "followup_report", "reported_symptoms": True},
    ]

    for record in records:
        event = FakeEvent(payload={"action": "ingest", "record": record})
        await agent.on_event(event, ctx)

    # Request aggregate
    event = FakeEvent(payload={"action": "aggregate"})
    response = await agent.on_event(event, ctx)

    aggregate = response.message["aggregate"]

    assert aggregate["total_records"] == 3
    assert aggregate["appointment_confirmed"] == 2
    assert aggregate["followup_reports"] == 1
    assert aggregate["symptom_reports"] == 1


@pytest.mark.asyncio
async def test_aggregate_empty(analytics_config, memory_bank, session):
    """Test aggregating with no records."""
    agent = AnalyticsAgent(analytics_config, memory_bank=memory_bank)

    event = FakeEvent(payload={"action": "aggregate"})
    ctx = FakeCtx(session)

    response = await agent.on_event(event, ctx)

    aggregate = response.message["aggregate"]
    assert aggregate["total_records"] == 0


@pytest.mark.asyncio
async def test_export_placeholder(analytics_config, memory_bank, session):
    """Test export placeholder functionality."""
    agent = AnalyticsAgent(analytics_config, memory_bank=memory_bank)
    ctx = FakeCtx(session)

    # Ingest some data
    record = {"event": "test_event"}
    event = FakeEvent(payload={"action": "ingest", "record": record})
    await agent.on_event(event, ctx)

    # Request export
    event = FakeEvent(payload={"action": "export"})
    response = await agent.on_event(event, ctx)

    exported = response.message["exported"]
    assert "exported_at" in exported
    assert exported["records_exported"] == 1


@pytest.mark.asyncio
async def test_metrics_tracking(analytics_config, memory_bank, session):
    """Test that analytics agent tracks its own metrics."""
    agent = AnalyticsAgent(analytics_config, memory_bank=memory_bank)
    ctx = FakeCtx(session)

    # Ingest records
    for i in range(5):
        record = {"event": f"event_{i}"}
        event = FakeEvent(payload={"action": "ingest", "record": record})
        await agent.on_event(event, ctx)

    # Check metrics
    snapshot = agent.metrics.snapshot()
    assert snapshot["counters"]["records_ingested"] == 5


def test_ingest_from_memorybank(analytics_config, memory_bank):
    """Test ingesting records from MemoryBank."""
    agent = AnalyticsAgent(analytics_config, memory_bank=memory_bank)

    # Add data to memory bank
    memory_bank.save("user1", {"event": "test1", "user_id": "user1", "email": "test@example.com"})
    memory_bank.save("user1", {"event": "test2", "user_id": "user1"})
    memory_bank.save("user2", {"event": "test3", "user_id": "user2"})

    # Ingest from memory bank
    agent.ingest_from_memorybank()

    # Should have sanitized records (user_id and email removed)
    assert len(agent._records) == 3
    for record in agent._records:
        assert "user_id" not in record
        assert "email" not in record
        assert "event" in record


def test_record_has_timestamp(analytics_config, memory_bank):
    """Test that ingested records get timestamped."""
    agent = AnalyticsAgent(analytics_config, memory_bank=memory_bank)

    record = {"event": "test_event"}
    agent._ingest(record)

    stored_record = agent._records[0]
    assert "_received_at" in stored_record
    # Should be valid ISO timestamp
    datetime.fromisoformat(stored_record["_received_at"])


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_followup_to_analytics_flow(followup_config, analytics_config, memory_bank, session):
    """Test data flow from FollowUp to Analytics."""
    followup_agent = FollowUpAgent(followup_config, memory_bank=memory_bank)
    analytics_agent = AnalyticsAgent(analytics_config, memory_bank=memory_bank)

    # Followup sends check-in
    event = FakeEvent(resume=True)
    ctx = FakeCtx(session)
    await followup_agent.on_event(event, ctx)

    # Analytics ingests from memory bank
    analytics_agent.ingest_from_memorybank()

    # Should have followup record
    assert len(analytics_agent._records) > 0
    assert any("followup_sent_at" in r for r in analytics_agent._records)


@pytest.mark.asyncio
async def test_concurrent_analytics_ingestion(analytics_config, memory_bank):
    """Test concurrent record ingestion."""
    agent = AnalyticsAgent(analytics_config, memory_bank=memory_bank)

    async def ingest_records(session_id):
        session = {"user_id": f"user_{session_id}"}
        ctx = FakeCtx(session)

        for i in range(10):
            record = {"event": f"event_{session_id}_{i}"}
            event = FakeEvent(payload={"action": "ingest", "record": record})
            await agent.on_event(event, ctx)

    # Run concurrent ingestions
    tasks = [ingest_records(i) for i in range(5)]
    await asyncio.gather(*tasks)

    # Should have all 50 records
    assert len(agent._records) == 50


@pytest.mark.asyncio
async def test_followup_schedule_and_checkin_cycle(followup_config, memory_bank, session):
    """Test complete schedule → pause → resume → checkin cycle."""
    agent = FollowUpAgent(followup_config, memory_bank=memory_bank)
    ctx = FakeCtx(session)

    # Step 1: Schedule reminder
    event1 = FakeEvent(resume=False)
    response1 = await agent.on_event(event1, ctx)

    assert hasattr(response1, 'pause_until')
    assert "followup_resume_at" in session

    # Step 2: Simulate resume (after pause)
    event2 = FakeEvent(resume=True)
    response2 = await agent.on_event(event2, ctx)

    assert hasattr(response2, 'text')
    assert "feeling" in response2.text.lower()

    # Step 3: Verify memory bank updated
    memories = memory_bank.get(session["user_id"])
    assert len(memories) > 0


@pytest.mark.asyncio
async def test_analytics_aggregate_statistics(analytics_config, memory_bank, session):
    """Test aggregate statistics calculations."""
    agent = AnalyticsAgent(analytics_config, memory_bank=memory_bank)
    ctx = FakeCtx(session)

    # Create diverse records
    records = [
        {"event": "appointment_confirmed", "clinic_id": "clinic_1"},
        {"event": "appointment_confirmed", "clinic_id": "clinic_2"},
        {"event": "appointment_confirmed", "clinic_id": "clinic_3"},
        {"event": "followup_report", "reported_symptoms": False},
        {"event": "followup_report", "reported_symptoms": True},
        {"event": "followup_report", "reported_symptoms": True},
    ]

    for record in records:
        event = FakeEvent(payload={"action": "ingest", "record": record})
        await agent.on_event(event, ctx)

    # Get aggregate
    event = FakeEvent(payload={"action": "aggregate"})
    response = await agent.on_event(event, ctx)

    aggregate = response.message["aggregate"]

    assert aggregate["total_records"] == 6
    assert aggregate["appointment_confirmed"] == 3
    assert aggregate["followup_reports"] == 3
    assert aggregate["symptom_reports"] == 2  # Only 2 with symptoms


@pytest.mark.asyncio
async def test_analytics_includes_metrics_in_aggregate(analytics_config, memory_bank, session):
    """Test that aggregate includes internal metrics."""
    agent = AnalyticsAgent(analytics_config, memory_bank=memory_bank)
    ctx = FakeCtx(session)

    # Ingest some records
    record = {"event": "test"}
    event = FakeEvent(payload={"action": "ingest", "record": record})
    await agent.on_event(event, ctx)

    # Get aggregate
    event = FakeEvent(payload={"action": "aggregate"})
    response = await agent.on_event(event, ctx)

    aggregate = response.message["aggregate"]

    # Should include metrics snapshot
    assert "metrics" in aggregate
    assert "counters" in aggregate["metrics"]
    assert "gauges" in aggregate["metrics"]


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_analytics_unknown_action(analytics_config, memory_bank, session):
    """Test analytics agent handles unknown actions."""
    agent = AnalyticsAgent(analytics_config, memory_bank=memory_bank)

    event = FakeEvent(payload={"action": "unknown_action"})
    ctx = FakeCtx(session)

    response = await agent.on_event(event, ctx)

    assert "error" in response.message
    assert "unknown action" in response.message["error"].lower()


@pytest.mark.asyncio
async def test_followup_without_user_id(followup_config, memory_bank):
    """Test followup agent handles missing user_id gracefully."""
    agent = FollowUpAgent(followup_config, memory_bank=memory_bank)

    session = {"user_id": None}  # Missing user_id
    ctx = FakeCtx(session)

    event = FakeEvent(resume=True)

    # Should not crash
    response = await agent.on_event(event, ctx)
    assert response is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])