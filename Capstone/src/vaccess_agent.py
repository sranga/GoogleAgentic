"""
VAccess Orchestrator - Production Version

Enhanced with:
- Full observability (structured logging, metrics, tracing)
- Robust error handling with specific exceptions
- Performance monitoring
- Health checks
- Circuit breaker pattern for external dependencies
- Request validation
- Async/await consistency
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

from sub_agents.vaccine_info_agent import VaccineInfoAgent
from sub_agents.clinic_finder_agent import ClinicFinderAgent
from sub_agents.appointment_agent import AppointmentAgent
from sub_agents.followup_agent import FollowUpAgent
from sub_agents.analytics_agent import AnalyticsAgent
from memory import InMemorySessionService, MemoryBank
from tools import save_confirmation_to_file
from observability import (
    get_logger,
    metrics,
    TraceContext,
    set_trace_context,
    monitor_performance,
    health_checker
)

logger = get_logger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class VAccessError(Exception):
    """Base exception for V-Access system."""
    pass


class SessionNotFoundError(VAccessError):
    """Session does not exist."""
    pass


class AgentExecutionError(VAccessError):
    """Agent execution failed."""
    pass


class ClinicSearchError(VAccessError):
    """Clinic search operation failed."""
    pass


class AppointmentBookingError(VAccessError):
    """Appointment booking failed."""
    pass


class ValidationError(VAccessError):
    """Input validation failed."""
    pass


# ============================================================================
# WORKFLOW STATES
# ============================================================================

class WorkflowState(Enum):
    """Tracks the current state of the user workflow."""
    INITIAL = "initial"
    EDUCATION = "education"
    CLINIC_SEARCH = "clinic_search"
    APPOINTMENT_BOOKING = "appointment_booking"
    FOLLOW_UP = "follow_up"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# CIRCUIT BREAKER (for external service calls)
# ============================================================================

class CircuitBreaker:
    """
    Simple circuit breaker to prevent cascading failures.
    Opens after N failures, closes after timeout.
    """

    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.is_open = False

    def record_success(self):
        """Record successful call."""
        self.failure_count = 0
        self.is_open = False

    def record_failure(self):
        """Record failed call."""
        import time
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(
                "Circuit breaker opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold
            )

    def can_execute(self) -> bool:
        """Check if circuit allows execution."""
        import time
        if not self.is_open:
            return True

        # Check if timeout has passed
        if self.last_failure_time:
            elapsed = time.time() - self.last_failure_time
            if elapsed > self.timeout_seconds:
                logger.info("Circuit breaker attempting to close")
                self.is_open = False
                self.failure_count = 0
                return True

        return False


# ============================================================================
# PRODUCTION ORCHESTRATOR
# ============================================================================

class VAccessOrchestrator:
    """
    Production-grade orchestrator with full observability and error handling.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}

        # Session & memory
        self.session_service = InMemorySessionService()
        self.memory_bank = MemoryBank()

        # Instantiate agents
        self.vaccine_info = VaccineInfoAgent(config, memory_bank=self.memory_bank)
        self.clinic_finder = ClinicFinderAgent(config)
        self.appointment_agent = AppointmentAgent(config)
        self.followup_agent = FollowUpAgent(config, memory_bank=self.memory_bank)
        self.analytics_agent = AnalyticsAgent(config, memory_bank=self.memory_bank)

        # Circuit breakers for external services
        self.clinic_search_breaker = CircuitBreaker()
        self.booking_breaker = CircuitBreaker()

        # Concurrency control
        self._session_locks: Dict[str, asyncio.Lock] = {}

        # Register health checks
        self._register_health_checks()

        logger.info(
            "VAccessOrchestrator initialized",
            model=config.get("model"),
            agents=["vaccine_info", "clinic_finder", "appointment", "followup", "analytics"]
        )

    def _register_health_checks(self):
        """Register health check functions."""

        def check_memory_bank():
            # Simple check that memory bank is accessible
            try:
                self.memory_bank.get("health_check_user")
                return True
            except Exception:
                return False

        def check_session_service():
            try:
                session = self.session_service.create_session("health_check")
                return session is not None
            except Exception:
                return False

        health_checker.register_check("memory_bank", check_memory_bank)
        health_checker.register_check("session_service", check_session_service)

    async def _get_session_lock(self, user_id: str) -> asyncio.Lock:
        """Get or create a lock for a specific session."""
        if user_id not in self._session_locks:
            self._session_locks[user_id] = asyncio.Lock()
        return self._session_locks[user_id]

    def validate_user_id(self, user_id: str):
        """Validate user ID format."""
        if not user_id or not isinstance(user_id, str):
            raise ValidationError("Invalid user_id: must be non-empty string")
        if len(user_id) > 128:
            raise ValidationError("Invalid user_id: too long (max 128 chars)")

    def start_session(self, user_id: str, initial_input: str) -> Dict[str, Any]:
        """
        Start a new user session.

        Args:
            user_id: Unique user identifier
            initial_input: User's initial message

        Returns:
            Session dictionary

        Raises:
            ValidationError: If inputs are invalid
        """
        self.validate_user_id(user_id)

        if not initial_input or len(initial_input) > 1000:
            raise ValidationError("Invalid initial_input")

        with monitor_performance("start_session"):
            session = self.session_service.create_session(user_id)
            session["workflow_state"] = WorkflowState.INITIAL.value
            session["created_at"] = datetime.utcnow().isoformat()
            session["history"].append({
                "role": "user",
                "text": initial_input,
                "timestamp": datetime.utcnow().isoformat()
            })

            metrics.counter("sessions_started")
            metrics.gauge("active_sessions", len(self.session_service._sessions))

            logger.info(
                "Session started",
                user_id=user_id[:8],  # Log only partial ID
                initial_input_length=len(initial_input)
            )

            return session

    async def run_education(self, session: Dict[str, Any], user_input: str) -> str:
        """
        Run education phase - answer vaccine questions.

        Args:
            session: User session
            user_input: User's question

        Returns:
            Agent response text

        Raises:
            AgentExecutionError: If agent execution fails
        """
        trace = get_current_trace_context()

        with monitor_performance("education", labels={"agent": "vaccine_info"}):
            try:
                if trace:
                    with trace.span("vaccine_info_agent", operation="education"):
                        response = await self.vaccine_info.emit(
                            {"text": user_input},
                            session=session
                        )
                else:
                    response = await self.vaccine_info.emit(
                        {"text": user_input},
                        session=session
                    )

                # Extract text from response
                msg = self._extract_message_text(response)

                # Update session state
                session["workflow_state"] = WorkflowState.EDUCATION.value
                session["history"].append({
                    "role": "assistant",
                    "text": msg,
                    "timestamp": datetime.utcnow().isoformat()
                })

                metrics.counter("education_queries", labels={"status": "success"})

                logger.info(
                    "Education query completed",
                    user_id=session.get("user_id", "unknown")[:8],
                    response_length=len(msg)
                )

                return msg

            except Exception as e:
                metrics.counter("education_queries", labels={"status": "error"})
                logger.error(
                    "Education query failed",
                    error=str(e),
                    user_id=session.get("user_id", "unknown")[:8]
                )
                raise AgentExecutionError(f"Education agent failed: {e}") from e

    async def find_and_schedule(
        self,
        session: Dict[str, Any],
        location_query: str
    ) -> Dict[str, Any]:
        """
        Full end-to-end find + schedule + follow-up orchestration.

        Args:
            session: User session
            location_query: Location for clinic search (zip code, address, etc.)

        Returns:
            Booking confirmation dict or error dict

        Raises:
            ClinicSearchError: If clinic search fails
            AppointmentBookingError: If booking fails
        """
        user_id = session.get("user_id")
        trace = get_current_trace_context()

        # Acquire session lock for thread safety
        lock = await self._get_session_lock(user_id)

        async with lock:
            try:
                # Step 1: Find clinics
                candidates = await self._find_clinics(session, location_query, trace)

                if not candidates:
                    session["workflow_state"] = WorkflowState.FAILED.value
                    metrics.counter("workflow_failures", labels={"stage": "clinic_search"})
                    return {
                        "confirmed": False,
                        "reason": "no_clinics_found",
                        "error": "No vaccination clinics found in your area"
                    }

                # Step 2: Book appointment
                confirmation = await self._book_appointment(session, candidates, trace)

                if not confirmation.get("confirmed"):
                    session["workflow_state"] = WorkflowState.FAILED.value
                    metrics.counter("workflow_failures", labels={"stage": "booking"})
                    return confirmation

                # Step 3: Schedule follow-up
                await self._schedule_followup(session, confirmation, trace)

                # Step 4: Record analytics
                await self._record_analytics(session, confirmation)

                session["workflow_state"] = WorkflowState.COMPLETED.value
                metrics.counter("workflows_completed")

                logger.info(
                    "Workflow completed successfully",
                    user_id=user_id[:8],
                    clinic_id=confirmation.get("clinic_id")
                )

                return confirmation

            except ClinicSearchError as e:
                logger.error("Clinic search failed", error=str(e), user_id=user_id[:8])
                session["workflow_state"] = WorkflowState.FAILED.value
                return {"confirmed": False, "reason": "clinic_search_error", "error": str(e)}

            except AppointmentBookingError as e:
                logger.error("Booking failed", error=str(e), user_id=user_id[:8])
                session["workflow_state"] = WorkflowState.FAILED.value
                return {"confirmed": False, "reason": "booking_error", "error": str(e)}

            except Exception as e:
                logger.exception("Unexpected error in workflow", error=str(e))
                session["workflow_state"] = WorkflowState.FAILED.value
                metrics.counter("workflow_failures", labels={"stage": "unknown"})
                return {"confirmed": False, "reason": "unexpected_error", "error": str(e)}

    async def _find_clinics(
        self,
        session: Dict[str, Any],
        location_query: str,
        trace: Optional[TraceContext]
    ) -> list:
        """Find clinics with circuit breaker protection."""

        # Check circuit breaker
        if not self.clinic_search_breaker.can_execute():
            logger.warning("Clinic search circuit breaker is open")
            raise ClinicSearchError("Clinic search service is temporarily unavailable")

        session["location_query"] = location_query
        session["workflow_state"] = WorkflowState.CLINIC_SEARCH.value

        logger.info(
            "Starting clinic search",
            location_query=location_query,
            user_id=session.get("user_id", "unknown")[:8]
        )

        try:
            with monitor_performance("clinic_search", labels={"method": "auto"}):
                if trace:
                    with trace.span("clinic_finder_agent", operation="search", location=location_query):
                        clinic_resp = await self.clinic_finder.emit(
                            {"location_query": location_query},
                            session=session
                        )
                else:
                    clinic_resp = await self.clinic_finder.emit(
                        {"location_query": location_query},
                        session=session
                    )

            # Extract candidates from response
            candidates = self._extract_candidates(clinic_resp, session)

            self.clinic_search_breaker.record_success()
            metrics.counter("clinic_searches", labels={"status": "success"})

            logger.info(
                "Clinic search completed",
                candidates_found=len(candidates),
                location=location_query
            )

            return candidates

        except Exception as e:
            self.clinic_search_breaker.record_failure()
            metrics.counter("clinic_searches", labels={"status": "error"})
            logger.error("Clinic search failed", error=str(e))
            raise ClinicSearchError(f"Failed to search for clinics: {e}") from e

    async def _book_appointment(
        self,
        session: Dict[str, Any],
        candidates: list,
        trace: Optional[TraceContext]
    ) -> Dict[str, Any]:
        """Book appointment with circuit breaker protection."""

        if not self.booking_breaker.can_execute():
            logger.warning("Booking circuit breaker is open")
            raise AppointmentBookingError("Booking service is temporarily unavailable")

        session["last_clinics"] = candidates
        session["workflow_state"] = WorkflowState.APPOINTMENT_BOOKING.value

        logger.info(
            "Starting appointment booking",
            candidate_count=len(candidates),
            user_id=session.get("user_id", "unknown")[:8]
        )

        try:
            with monitor_performance("appointment_booking"):
                if trace:
                    with trace.span("appointment_agent", operation="book"):
                        appointment_result = await self.appointment_agent.emit({}, session=session)
                else:
                    appointment_result = await self.appointment_agent.emit({}, session=session)

            # Extract confirmation
            confirmation = self._extract_confirmation(appointment_result)

            if confirmation.get("confirmed"):
                # Save confirmation artifact
                try:
                    fname = save_confirmation_to_file(session["user_id"], confirmation)
                    logger.info("Confirmation saved", filename=fname)
                except Exception as e:
                    logger.warning("Failed to save confirmation file", error=str(e))

                self.booking_breaker.record_success()
                metrics.counter("appointments_booked", labels={"status": "success"})

                logger.info(
                    "Appointment booked successfully",
                    confirmation_id=confirmation.get("confirmation_id"),
                    clinic_id=confirmation.get("clinic_id")
                )
            else:
                self.booking_breaker.record_failure()
                metrics.counter("appointments_booked", labels={"status": "failed"})

            return confirmation

        except Exception as e:
            self.booking_breaker.record_failure()
            metrics.counter("appointments_booked", labels={"status": "error"})
            logger.error("Appointment booking failed", error=str(e))
            raise AppointmentBookingError(f"Failed to book appointment: {e}") from e

    async def _schedule_followup(
        self,
        session: Dict[str, Any],
        confirmation: Dict[str, Any],
        trace: Optional[TraceContext]
    ):
        """Schedule follow-up reminder."""
        session["workflow_state"] = WorkflowState.FOLLOW_UP.value

        logger.info("Scheduling follow-up", user_id=session.get("user_id", "unknown")[:8])

        try:
            if trace:
                with trace.span("followup_agent", operation="schedule"):
                    await self.followup_agent.emit({}, session=session)
            else:
                await self.followup_agent.emit({}, session=session)

            metrics.counter("followups_scheduled")
            logger.info("Follow-up scheduled successfully")

        except Exception as e:
            logger.warning("Follow-up scheduling failed (non-critical)", error=str(e))
            # Don't fail the whole workflow if follow-up fails
            metrics.counter("followup_failures")

    async def _record_analytics(self, session: Dict[str, Any], confirmation: Dict[str, Any]):
        """Record anonymized analytics."""
        anon_record = {
            "event": "appointment_confirmed",
            "clinic_id": confirmation.get("clinic_id"),
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_duration_ms": self._calculate_workflow_duration(session),
        }

        try:
            await self.analytics_agent.emit(
                {"action": "ingest", "record": anon_record},
                session=None
            )
            metrics.counter("analytics_records_ingested")
        except Exception as e:
            logger.warning("Analytics recording failed (non-critical)", error=str(e))

    def _calculate_workflow_duration(self, session: Dict[str, Any]) -> float:
        """Calculate total workflow duration in milliseconds."""
        if "created_at" not in session:
            return 0.0

        created = datetime.fromisoformat(session["created_at"])
        duration = (datetime.utcnow() - created).total_seconds() * 1000
        return duration

    def _extract_message_text(self, response: Any) -> str:
        """Extract text from various response formats."""
        if hasattr(response, "text"):
            return str(response.text)
        elif hasattr(response, "message"):
            msg = response.message
            if isinstance(msg, dict) and "text" in msg:
                return str(msg["text"])
            return str(msg)
        return str(response)

    def _extract_candidates(self, response: Any, session: Dict[str, Any]) -> list:
        """Extract clinic candidates from response."""
        try:
            if hasattr(response, "message") and isinstance(response.message, dict):
                candidates = response.message.get("candidates")
                if candidates:
                    return candidates
        except Exception:
            pass

        # Fallback: check session
        return session.get("last_clinics", [])

    def _extract_confirmation(self, response: Any) -> Dict[str, Any]:
        """Extract confirmation from appointment response."""
        if hasattr(response, "message") and isinstance(response.message, dict):
            return response.message
        elif isinstance(response, dict):
            return response

        return {"confirmed": False, "reason": "invalid_response"}

    async def run_demo_flow(self, user_id: str, location_query: str) -> Dict[str, Any]:
        """
        Run a complete demo workflow with full tracing.

        Args:
            user_id: User identifier
            location_query: Location for clinic search

        Returns:
            Dict with session and confirmation data
        """
        trace = TraceContext()
        set_trace_context(trace)

        with trace.span("demo_workflow", user_id=user_id[:8]):
            # 1) Start session
            session = self.start_session(user_id, "Hi, I need a vaccine")

            # 2) Education
            await self.run_education(session, "What vaccines are available?")

            # 3) Find + Schedule
            confirmation = await self.find_and_schedule(session, location_query)

            # Log trace summary
            summary = trace.get_trace_summary()
            logger.info(
                "Demo workflow completed",
                trace_id=trace.trace_id,
                total_duration_ms=summary["total_duration_ms"],
                confirmed=confirmation.get("confirmed")
            )

            return {
                "session": session,
                "confirmation": confirmation,
                "trace": summary
            }