"""
Unit tests for VAccessOrchestrator

Tests:
- Sequential orchestration
- Error propagation
- Session locking
- Workflow state transitions
- Circuit breaker integration
- Health checks
- Metrics tracking
- Concurrent session handling

All tests use mock agents from conftest.py to avoid Pydantic patching issues.
"""

import pytest
import asyncio

from vaccess_agent import (
    VAccessOrchestrator,
    WorkflowState,
    VAccessError,
    ClinicSearchError,
    AppointmentBookingError,
    ValidationError,
    CircuitBreaker,
)
from config import CONFIG

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def config():
    """Default configuration."""
    return CONFIG.copy()


@pytest.fixture
def orchestrator(config):
    """Create orchestrator instance."""
    return VAccessOrchestrator(config)


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

def test_orchestrator_initialization(config):
    """Test orchestrator initializes correctly."""
    orch = VAccessOrchestrator(config)

    assert orch.vaccine_info is not None
    assert orch.clinic_finder is not None
    assert orch.appointment_agent is not None
    assert orch.followup_agent is not None
    assert orch.analytics_agent is not None
    assert orch.session_service is not None
    assert orch.memory_bank is not None


def test_orchestrator_has_circuit_breakers(orchestrator):
    """Test orchestrator has circuit breakers configured."""
    assert orchestrator.clinic_search_breaker is not None
    assert orchestrator.booking_breaker is not None
    assert isinstance(orchestrator.clinic_search_breaker, CircuitBreaker)
    assert isinstance(orchestrator.booking_breaker, CircuitBreaker)


def test_health_checks_registered(orchestrator):
    """Test health checks are registered."""
    from observability import health_checker

    # Health checker should have checks registered
    # This is a basic check - actual health check tests below
    assert len(health_checker._checks) > 0


# ============================================================================
# SESSION MANAGEMENT TESTS
# ============================================================================

def test_start_session_valid(orchestrator):
    """Test starting a session with valid inputs."""
    session = orchestrator.start_session("user123", "Hello, I need help")

    assert session is not None
    assert session["user_id"] == "user123"
    assert session["workflow_state"] == WorkflowState.INITIAL.value
    assert "created_at" in session
    assert len(session["history"]) == 1
    assert session["history"][0]["role"] == "user"


def test_start_session_invalid_user_id(orchestrator):
    """Test starting session with invalid user_id."""
    with pytest.raises(ValidationError):
        orchestrator.start_session("ab", "Hello")  # Too short

    with pytest.raises(ValidationError):
        orchestrator.start_session("user@invalid", "Hello")  # Invalid chars


def test_start_session_invalid_input(orchestrator):
    """Test starting session with invalid initial input."""
    with pytest.raises(ValidationError):
        orchestrator.start_session("user123", "")  # Empty

    with pytest.raises(ValidationError):
        long_input = "a" * 1001
        orchestrator.start_session("user123", long_input)  # Too long


def test_validate_user_id(orchestrator):
    """Test user ID validation."""
    # Valid IDs
    orchestrator.validate_user_id("user123")
    orchestrator.validate_user_id("user_test")
    orchestrator.validate_user_id("user-123-abc")

    # Invalid IDs
    with pytest.raises(ValidationError):
        orchestrator.validate_user_id("")

    with pytest.raises(ValidationError):
        orchestrator.validate_user_id("a" * 129)


# ============================================================================
# EDUCATION PHASE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_run_education_success(orchestrator, mock_vaccine_info):
    """Test successful education phase."""
    session = orchestrator.start_session("user_edu", "Hello")

    mock_vaccine_info.response_text = "Vaccines help prevent diseases."

    original = orchestrator.vaccine_info
    orchestrator.vaccine_info = mock_vaccine_info

    try:
        response = await orchestrator.run_education(session, "What is a vaccine?")

        assert isinstance(response, str)
        assert len(response) > 0
        assert session["workflow_state"] == WorkflowState.EDUCATION.value
        assert len(session["history"]) >= 2  # User + assistant messages
    finally:
        orchestrator.vaccine_info = original


@pytest.mark.asyncio
async def test_run_education_updates_session(orchestrator, mock_vaccine_info):
    """Test education phase updates session history."""
    session = orchestrator.start_session("user_edu2", "Hello")
    initial_history_len = len(session["history"])

    original = orchestrator.vaccine_info
    orchestrator.vaccine_info = mock_vaccine_info

    try:
        await orchestrator.run_education(session, "Tell me about side effects")

        # Should have 2 new entries (user question + assistant response)
        assert len(session["history"]) == initial_history_len + 2
    finally:
        orchestrator.vaccine_info = original


@pytest.mark.asyncio
async def test_run_education_error_handling(orchestrator, mock_vaccine_info):
    """Test education phase error handling."""
    session = orchestrator.start_session("user_edu3", "Hello")

    # Configure mock to raise error
    mock_vaccine_info.should_raise_error = True
    mock_vaccine_info.error_message = "Test error"

    original = orchestrator.vaccine_info
    orchestrator.vaccine_info = mock_vaccine_info

    try:
        with pytest.raises(Exception, match="Test error"):
            await orchestrator.run_education(session, "Test question")
    finally:
        orchestrator.vaccine_info = original


# ============================================================================
# WORKFLOW STATE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_workflow_state_progression(orchestrator, mock_vaccine_info):
    """Test workflow state progresses correctly."""
    session = orchestrator.start_session("user_wf", "Hello")

    # Initial state
    assert session["workflow_state"] == WorkflowState.INITIAL.value

    original = orchestrator.vaccine_info
    orchestrator.vaccine_info = mock_vaccine_info

    try:
        # After education
        await orchestrator.run_education(session, "What are vaccines?")
        assert session["workflow_state"] == WorkflowState.EDUCATION.value
    finally:
        orchestrator.vaccine_info = original


# ============================================================================
# CIRCUIT BREAKER TESTS
# ============================================================================

def test_circuit_breaker_initialization():
    """Test circuit breaker initializes correctly."""
    breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=30)

    assert breaker.failure_threshold == 3
    assert breaker.timeout_seconds == 30
    assert breaker.failure_count == 0
    assert breaker.is_open is False


def test_circuit_breaker_opens_after_failures():
    """Test circuit breaker opens after threshold failures."""
    breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=30)

    # Record failures
    breaker.record_failure()
    assert breaker.can_execute() is True  # Still closed

    breaker.record_failure()
    assert breaker.can_execute() is True  # Still closed

    breaker.record_failure()
    assert breaker.is_open is True
    assert breaker.can_execute() is False  # Now open


def test_circuit_breaker_closes_after_timeout():
    """Test circuit breaker closes after timeout."""
    breaker = CircuitBreaker(failure_threshold=2, timeout_seconds=1)

    # Open the breaker
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_open is True

    # Wait for timeout
    import time
    time.sleep(1.5)

    # Should close
    assert breaker.can_execute() is True


def test_circuit_breaker_resets_on_success():
    """Test circuit breaker resets failure count on success."""
    breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=30)

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.failure_count == 2

    breaker.record_success()
    assert breaker.failure_count == 0
    assert breaker.is_open is False


# ============================================================================
# CONCURRENT SESSION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_concurrent_sessions(orchestrator, mock_vaccine_info):
    """Test handling multiple concurrent sessions."""
    # Create multiple sessions
    sessions = []
    for i in range(5):
        session = orchestrator.start_session(f"user_{i}", f"Hello {i}")
        sessions.append(session)

    original = orchestrator.vaccine_info
    orchestrator.vaccine_info = mock_vaccine_info

    try:
        # Run education concurrently
        tasks = [
            orchestrator.run_education(sess, "What is a vaccine?")
            for sess in sessions
        ]
        responses = await asyncio.gather(*tasks)

        # All should succeed
        assert len(responses) == 5
        assert all(isinstance(r, str) for r in responses)
        assert all(len(r) > 0 for r in responses)
    finally:
        orchestrator.vaccine_info = original


@pytest.mark.asyncio
async def test_session_locking(orchestrator):
    """Test session-level locking prevents race conditions."""
    session = orchestrator.start_session("user_lock", "Hello")

    # Get lock for this session
    lock = await orchestrator._get_session_lock("user_lock")

    assert lock is not None

    # Same user should get same lock
    lock2 = await orchestrator._get_session_lock("user_lock")
    assert lock is lock2

    # Different user should get different lock
    lock3 = await orchestrator._get_session_lock("user_other")
    assert lock is not lock3


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_clinic_search_error_handling(orchestrator, mock_clinic_finder):
    """Test error handling in clinic search."""
    session = orchestrator.start_session("user_err", "Hello")

    # Configure mock to raise error
    mock_clinic_finder.should_raise_error = True
    mock_clinic_finder.error_message = "Search failed"

    original = orchestrator.clinic_finder
    orchestrator.clinic_finder = mock_clinic_finder

    try:
        result = await orchestrator.find_and_schedule(session, "94110")

        # Should return error dict instead of crashing
        assert result["confirmed"] is False
        assert "error" in result or "reason" in result
    finally:
        orchestrator.clinic_finder = original


@pytest.mark.asyncio
async def test_appointment_booking_error_handling(orchestrator, mock_appointment_agent):
    """Test error handling in appointment booking."""
    session = orchestrator.start_session("user_book_err", "Hello")

    # Configure mock to raise error
    mock_appointment_agent.should_raise_error = True
    mock_appointment_agent.error_message = "Booking failed"

    original = orchestrator.appointment_agent
    orchestrator.appointment_agent = mock_appointment_agent

    try:
        result = await orchestrator.find_and_schedule(session, "94110")
        assert result["confirmed"] is False
    finally:
        orchestrator.appointment_agent = original


# ============================================================================
# HELPER METHOD TESTS
# ============================================================================

def test_extract_message_text(orchestrator):
    """Test extracting text from various response formats."""

    # Test with .text attribute
    class MockResponse:
        text = "Test message"

    text = orchestrator._extract_message_text(MockResponse())
    assert text == "Test message"

    # Test with .message dict
    class MockResponse2:
        message = {"text": "Test message 2"}

    text = orchestrator._extract_message_text(MockResponse2())
    assert text == "Test message 2"

    # Test with string
    text = orchestrator._extract_message_text("Direct string")
    assert text == "Direct string"


def test_calculate_workflow_duration(orchestrator):
    """Test workflow duration calculation."""
    session = orchestrator.start_session("user_duration", "Hello")

    # Should be very small (just created)
    duration = orchestrator._calculate_workflow_duration(session)
    assert duration >= 0
    assert duration < 1000  # Less than 1 second


def test_calculate_workflow_duration_no_timestamp(orchestrator):
    """Test workflow duration with missing timestamp."""
    session = {"user_id": "test"}  # No created_at

    duration = orchestrator._calculate_workflow_duration(session)
    assert duration == 0.0


# ============================================================================
# HEALTH CHECK TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_health_check_success(orchestrator):
    """Test health check returns healthy status."""
    from observability import health_checker

    status = await health_checker.check_health()

    assert "status" in status
    assert "checks" in status
    # At least memory_bank and session_service checks
    assert len(status["checks"]) >= 2


@pytest.mark.asyncio
async def test_health_check_individual_checks(orchestrator):
    """Test individual health checks."""
    from observability import health_checker

    status = await health_checker.check_health()

    # Check that our registered checks exist
    checks = status["checks"]
    assert "memory_bank" in checks
    assert "session_service" in checks


# ============================================================================
# METRICS TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_metrics_tracked_on_session_start(orchestrator):
    """Test metrics are tracked when session starts."""
    from observability import metrics

    initial_snapshot = metrics.snapshot()
    initial_count = initial_snapshot["counters"].get("sessions_started", 0)

    orchestrator.start_session("user_metrics", "Hello")

    new_snapshot = metrics.snapshot()
    new_count = new_snapshot["counters"].get("sessions_started", 0)

    assert new_count > initial_count


@pytest.mark.asyncio
async def test_metrics_tracked_on_education(orchestrator, mock_vaccine_info):
    """Test metrics are tracked during education."""
    from observability import metrics

    session = orchestrator.start_session("user_edu_metrics", "Hello")

    initial_snapshot = metrics.snapshot()
    initial_count = initial_snapshot["counters"].get("education_queries", 0)

    original = orchestrator.vaccine_info
    orchestrator.vaccine_info = mock_vaccine_info

    try:
        await orchestrator.run_education(session, "What is a vaccine?")

        new_snapshot = metrics.snapshot()
        new_count = new_snapshot["counters"].get("education_queries", 0)

        assert new_count > initial_count
    finally:
        orchestrator.vaccine_info = original


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_full_demo_flow(orchestrator, replace_orchestrator_agents, all_mock_agents):
    """Test complete demo workflow."""

    with replace_orchestrator_agents(orchestrator, all_mock_agents):
        result = await orchestrator.run_demo_flow("user_demo", "94110")

        assert "session" in result
        assert "confirmation" in result
        assert "trace" in result

        # Session should exist
        assert result["session"]["user_id"] == "user_demo"

        # Should have trace summary
        trace = result["trace"]
        assert "trace_id" in trace
        assert "total_duration_ms" in trace


@pytest.mark.asyncio
async def test_workflow_state_on_failure(orchestrator, replace_orchestrator_agents,
                                         all_mock_agents):
    """Test workflow state set to FAILED on error."""
    session = orchestrator.start_session("user_fail", "Hello")

    # Configure mock to fail
    all_mock_agents["clinic_finder"].should_raise_error = True
    all_mock_agents["clinic_finder"].error_message = "Forced failure"

    with replace_orchestrator_agents(orchestrator, all_mock_agents):
        result = await orchestrator.find_and_schedule(session, "94110")

        # State should be FAILED
        assert session["workflow_state"] == WorkflowState.FAILED.value
        assert result["confirmed"] is False


@pytest.mark.asyncio
async def test_workflow_state_on_success(orchestrator, replace_orchestrator_agents,
                                         all_mock_agents):
    """Test workflow state set to COMPLETED on success."""
    session = orchestrator.start_session("user_success", "Hello")

    # All mocks are already configured for success by default

    with replace_orchestrator_agents(orchestrator, all_mock_agents):
        result = await orchestrator.find_and_schedule(session, "94110")

        # Should complete successfully
        assert result["confirmed"] is True
        assert session["workflow_state"] == WorkflowState.COMPLETED.value


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_empty_clinic_list(orchestrator, replace_orchestrator_agents,
                                 all_mock_agents):
    """Test handling of empty clinic list."""
    session = orchestrator.start_session("user_no_clinics", "Hello")

    # Configure mock to return no candidates
    all_mock_agents["clinic_finder"].candidates = []

    with replace_orchestrator_agents(orchestrator, all_mock_agents):
        result = await orchestrator.find_and_schedule(session, "94110")

        assert result["confirmed"] is False
        assert result["reason"] == "no_clinics_found"


@pytest.mark.asyncio
async def test_multiple_workflow_runs_same_user(orchestrator, mock_vaccine_info):
    """Test running multiple workflows for same user."""
    session = orchestrator.start_session("user_multi", "Hello")

    original = orchestrator.vaccine_info
    orchestrator.vaccine_info = mock_vaccine_info

    try:
        # Run education multiple times
        await orchestrator.run_education(session, "Question 1")
        await orchestrator.run_education(session, "Question 2")
        await orchestrator.run_education(session, "Question 3")

        # Should have accumulated history
        assert len(session["history"]) >= 7  # Initial + 3*(user+assistant)
    finally:
        orchestrator.vaccine_info = original


@pytest.mark.asyncio
async def test_orchestrator_with_tracing(orchestrator, mock_vaccine_info):
    """Test orchestrator with distributed tracing."""
    from observability import TraceContext, set_trace_context

    trace = TraceContext()
    set_trace_context(trace)

    session = orchestrator.start_session("user_trace", "Hello")

    original = orchestrator.vaccine_info
    orchestrator.vaccine_info = mock_vaccine_info

    try:
        await orchestrator.run_education(session, "What is a vaccine?")

        # Should have trace spans
        summary = trace.get_trace_summary()
        assert summary["span_count"] > 0
    finally:
        orchestrator.vaccine_info = original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])