"""
V-Access Orchestrator

Central coordinator for the vaccine access multi-agent system.
Manages the complete vaccination workflow from education to follow-up.

Features:
- Sequential orchestration of specialized agents
- Session and memory management
- Circuit breaker pattern for fault tolerance
- Full observability (logging, metrics, tracing)
- Workflow state tracking
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
from observability import get_logger, metrics, TraceContext, set_trace_context, health_checker
from security import InputValidator, ValidationError, SecureStorage

logger = get_logger(__name__)


class WorkflowState(Enum):
    """Tracks the current state of the user workflow."""
    INITIAL = "initial"
    EDUCATION = "education"
    CLINIC_SEARCH = "clinic_search"
    APPOINTMENT_BOOKING = "appointment_booking"
    FOLLOW_UP = "follow_up"
    COMPLETED = "completed"
    FAILED = "failed"


class VAccessError(Exception):
    """Base exception for V-Access system."""
    pass


class ClinicSearchError(VAccessError):
    """Clinic search operation failed."""
    pass


class AppointmentBookingError(VAccessError):
    """Appointment booking failed."""
    pass


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.
    Opens after threshold failures, closes after timeout.
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

        if self.last_failure_time:
            elapsed = time.time() - self.last_failure_time
            if elapsed > self.timeout_seconds:
                logger.info("Circuit breaker attempting to close")
                self.is_open = False
                self.failure_count = 0
                return True

        return False


class VAccessOrchestrator:
    """
    Production-grade orchestrator for the vaccine access workflow.
    Coordinates all specialized agents through the complete user journey.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}

        # Session and memory services
        self.session_service = InMemorySessionService()
        self.memory_bank = MemoryBank()

        # Initialize specialized agents
        self.vaccine_info = VaccineInfoAgent(config, memory_bank=self.memory_bank)
        self.clinic_finder = ClinicFinderAgent(config)
        self.appointment_agent = AppointmentAgent(config)
        self.followup_agent = FollowUpAgent(config, memory_bank=self.memory_bank)
        self.analytics_agent = AnalyticsAgent(config, memory_bank=self.memory_bank)

        # Circuit breakers for external service calls
        self.clinic_search_breaker = CircuitBreaker()
        self.booking_breaker = CircuitBreaker()

        # Secure storage for confirmations
        self.secure_storage = SecureStorage()

        # Session locks for concurrency control
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
        user_id = InputValidator.validate_user_id(user_id)

        if not initial_input or len(initial_input) > 1000:
            raise ValidationError("Invalid initial_input")

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

        logger.info("Session started", user_id=user_id[:8], initial_input_length=len(initial_input))

        return session

    async def run_education(self, session: Dict[str, Any], user_input: str) -> str:
        """
        Run education phase - answer vaccine questions.

        Args:
            session: User session
            user_input: User's question

        Returns:
            Agent response text
        """
        try:
            response = await self.vaccine_info.emit({"text": user_input}, session=session)

            msg = response.text if hasattr(response, "text") else str(response)

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
            logger.error("Education query failed", error=str(e))
            raise

    async def find_and_schedule(self, session: Dict[str, Any], location_query: str) -> Dict[str, Any]:
        """
        Complete workflow: find clinics + book appointment + schedule follow-up.

        Args:
            session: User session
            location_query: Location for clinic search

        Returns:
            Booking confirmation or error dictionary
        """
        user_id = session.get("user_id")
        lock = await self._get_session_lock(user_id)

        async with lock:
            try:
                # Step 1: Find clinics
                candidates = await self._find_clinics(session, location_query)

                if not candidates:
                    session["workflow_state"] = WorkflowState.FAILED.value
                    metrics.counter("workflow_failures", labels={"stage": "clinic_search"})
                    return {
                        "confirmed": False,
                        "reason": "no_clinics_found",
                        "error": "No vaccination clinics found in your area"
                    }

                # Step 2: Book appointment
                confirmation = await self._book_appointment(session, candidates)

                if not confirmation.get("confirmed"):
                    session["workflow_state"] = WorkflowState.FAILED.value
                    metrics.counter("workflow_failures", labels={"stage": "booking"})
                    return confirmation

                # Step 3: Schedule follow-up
                await self._schedule_followup(session, confirmation)

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
                logger.error("Clinic search failed", error=str(e))
                session["workflow_state"] = WorkflowState.FAILED.value
                return {"confirmed": False, "reason": "clinic_search_error", "error": str(e)}

            except AppointmentBookingError as e:
                logger.error("Booking failed", error=str(e))
                session["workflow_state"] = WorkflowState.FAILED.value
                return {"confirmed": False, "reason": "booking_error", "error": str(e)}

            except Exception as e:
                logger.exception("Unexpected workflow error", error=str(e))
                session["workflow_state"] = WorkflowState.FAILED.value
                metrics.counter("workflow_failures", labels={"stage": "unknown"})
                return {"confirmed": False, "reason": "unexpected_error", "error": str(e)}

    async def _find_clinics(self, session: Dict[str, Any], location_query: str) -> list:
        """Find clinics with circuit breaker protection."""
        if not self.clinic_search_breaker.can_execute():
            raise ClinicSearchError("Clinic search service temporarily unavailable")

        session["location_query"] = location_query
        session["workflow_state"] = WorkflowState.CLINIC_SEARCH.value

        logger.info("Starting clinic search", location_query=location_query)

        try:
            clinic_resp = await self.clinic_finder.emit(
                {"location_query": location_query},
                session=session
            )

            candidates = clinic_resp.get("candidates", []) if isinstance(clinic_resp, dict) else []
            if not candidates:
                candidates = session.get("last_clinics", [])

            self.clinic_search_breaker.record_success()
            metrics.counter("clinic_searches", labels={"status": "success"})

            logger.info("Clinic search completed", candidates_found=len(candidates))

            return candidates

        except Exception as e:
            self.clinic_search_breaker.record_failure()
            metrics.counter("clinic_searches", labels={"status": "error"})
            raise ClinicSearchError(f"Failed to search for clinics: {e}") from e

    async def _book_appointment(self, session: Dict[str, Any], candidates: list) -> Dict[str, Any]:
        """Book appointment with circuit breaker protection."""
        if not self.booking_breaker.can_execute():
            raise AppointmentBookingError("Booking service temporarily unavailable")

        session["last_clinics"] = candidates
        session["workflow_state"] = WorkflowState.APPOINTMENT_BOOKING.value

        logger.info("Starting appointment booking", candidate_count=len(candidates))

        try:
            result = await self.appointment_agent.emit({}, session=session)

            if isinstance(result, dict) and result.get("confirmed"):
                # Save confirmation securely
                try:
                    filepath = self.secure_storage.save_confirmation(
                        session["user_id"],
                        result
                    )
                    result["confirmation_file"] = filepath
                    logger.info("Confirmation saved", filename=filepath)
                except Exception as e:
                    logger.warning("Failed to save confirmation", error=str(e))

                self.booking_breaker.record_success()
                metrics.counter("appointments_booked", labels={"status": "success"})

                return result

            self.booking_breaker.record_failure()
            metrics.counter("appointments_booked", labels={"status": "failed"})
            return {"confirmed": False, "reason": "booking_not_confirmed"}

        except Exception as e:
            self.booking_breaker.record_failure()
            metrics.counter("appointments_booked", labels={"status": "error"})
            raise AppointmentBookingError(f"Failed to book appointment: {e}") from e

    async def _schedule_followup(self, session: Dict[str, Any], confirmation: Dict[str, Any]):
        """Schedule follow-up reminder."""
        session["workflow_state"] = WorkflowState.FOLLOW_UP.value

        logger.info("Scheduling follow-up", user_id=session.get("user_id", "unknown")[:8])

        try:
            await self.followup_agent.emit({}, session=session)
            metrics.counter("followups_scheduled")
            logger.info("Follow-up scheduled")
        except Exception as e:
            logger.warning("Follow-up scheduling failed (non-critical)", error=str(e))
            metrics.counter("followup_failures")

    async def _record_analytics(self, session: Dict[str, Any], confirmation: Dict[str, Any]):
        """Record anonymized analytics."""
        anon_record = {
            "event": "appointment_confirmed",
            "clinic_id": confirmation.get("clinic_id"),
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            await self.analytics_agent.emit(
                {"action": "ingest", "record": anon_record},
                session=None
            )
            metrics.counter("analytics_records_ingested")
        except Exception as e:
            logger.warning("Analytics recording failed (non-critical)", error=str(e))

    async def run_demo_flow(self, user_id: str, location_query: str) -> Dict[str, Any]:
        """
        Run a complete demo workflow with full tracing.

        Args:
            user_id: User identifier
            location_query: Location for clinic search

        Returns:
            Dictionary with session, confirmation, and trace data
        """
        trace = TraceContext()
        set_trace_context(trace)

        with trace.span("demo_workflow", user_id=user_id[:8]):
            session = self.start_session(user_id, "Hi, I need a vaccine")

            await self.run_education(session, "What vaccines are available?")

            confirmation = await self.find_and_schedule(session, location_query)

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