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
import contextlib
from typing import Dict, Any, Optional
from datetime import datetime, UTC
from enum import Enum

from sub_agents.vaccine_info_agent import VaccineInfoAgent
from sub_agents.clinic_finder_agent import ClinicFinderAgent
from sub_agents.appointment_agent import AppointmentAgent
from sub_agents.followup_agent import FollowUpAgent
from sub_agents.analytics_agent import AnalyticsAgent
from memory import InMemorySessionService, MemoryBank
from tools import save_confirmation_to_file
from observability import get_logger, metrics, TraceContext, set_trace_context, health_checker, get_current_trace_context
from security import InputValidator, ValidationError, SecureStorage

from google.adk.agents import Agent
from google.adk.events import EventActions
from pydantic import PrivateAttr

"""
# Use the try-except block to handle testing environment for ADK types
try:
    from google.adk.agents import Agent as AdkBaseAgent
    from google.adk.events import EventActions
except ImportError:
    # Minimal mock for testing orchestrator flow without full ADK environment
    class AdkBaseAgent:
        def __init__(self, *args, **kwargs):
            pass

    class EventActions:
        def __init__(self, state_delta=None, resume=False, **kwargs):
            # This is the minimal set of attributes the orchestrator accesses
            self.state_delta = state_delta if state_delta is not None else {}
            self.resume = resume
            for k, v in kwargs.items():
                setattr(self, k, v)
"""

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
                extra={
                    "failure_count": self.failure_count,
                    "threshold": self.failure_threshold
                }
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


class VAccessOrchestrator(Agent):
    """
    Production-grade orchestrator for the vaccine access workflow.
    Coordinates all specialized agents through the complete user journey.
    """
    _config: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _session_service: Any = PrivateAttr(default=None)
    _memory_bank: Any = PrivateAttr(default=None)
    _vaccine_info: Any = PrivateAttr(default=None)
    _clinic_finder: Any = PrivateAttr(default=None)
    _appointment_agent: Any = PrivateAttr(default=None)
    _followup_agent: Any = PrivateAttr(default=None)
    _analytics_agent: Any = PrivateAttr(default=None)
    _clinic_search_breaker: Any = PrivateAttr(default=None)
    _booking_breaker: Any = PrivateAttr(default=None)
    _secure_storage: Any = PrivateAttr(default=None)
    _session_locks: Dict[str, asyncio.Lock] = PrivateAttr(default_factory=dict)

    def __init__(self, config: Dict[str, Any]):
        # 1. Call the parent constructor (BaseAgent/Agent) and set the agent's name.
        # This registers the agent's identity for the Web UI.
        try:
            super().__init__(
            name="VAccessOrchestrator",
            model=config.get("model", "gemini-2.0-flash"),
            description="Main orchestrator for vaccine access workflow",
            instruction="Coordinate vaccination workflow from education to follow-up"
        )
        except TypeError:
            # Fallback for environments where BaseAgent has no __init__
            pass

        object.__setattr__(self, '_config', config or {})

        # Session and memory services
        object.__setattr__(self, '_session_service', InMemorySessionService())
        object.__setattr__(self, '_memory_bank', MemoryBank())

        # Initialize specialized agents
        object.__setattr__(self, '_vaccine_info', VaccineInfoAgent(config, memory_bank=self.memory_bank))
        object.__setattr__(self, '_clinic_finder', ClinicFinderAgent(config))
        object.__setattr__(self, '_appointment_agent', AppointmentAgent(config))
        object.__setattr__(self, '_followup_agent', FollowUpAgent(config, memory_bank=self.memory_bank))
        object.__setattr__(self, '_analytics_agent', AnalyticsAgent(config, memory_bank=self.memory_bank))

        # Circuit breakers for external service calls
        object.__setattr__(self, '_clinic_search_breaker', CircuitBreaker())
        object.__setattr__(self, '_booking_breaker', CircuitBreaker())

        # Secure storage for confirmations
        object.__setattr__(self, '_secure_storage', SecureStorage())

        # Session locks for concurrency control
        object.__setattr__(self, '_session_locks', {})

        # Register health checks
        self._register_health_checks()

        logger.info(
            "VAccessOrchestrator initialized",
            extra={
                "model": config.get("model"),
                "agents": ["vaccine_info", "clinic_finder", "appointment", "followup", "analytics"]
            }
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

    def validate_user_id(self, user_id: str) -> str:
        """
        Validate user ID format.

        Args:
            user_id: User identifier to validate

        Returns:
            Validated user_id

        Raises:
            ValidationError: If user_id is invalid
        """
        return InputValidator.validate_user_id(user_id)

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
        user_id = self.validate_user_id(user_id)

        if not initial_input or len(initial_input) > 1000:
            raise ValidationError("Invalid initial_input")

        session = self.session_service.create_session(user_id)
        session["workflow_state"] = WorkflowState.INITIAL.value
        session["created_at"] = datetime.now(UTC).isoformat()
        session["history"].append({
            "role": "user",
            "text": initial_input,
            "timestamp": datetime.now(UTC).isoformat()
        })

        metrics.counter("sessions_started")
        metrics.gauge("active_sessions", len(self.session_service._sessions))

        logger.info(
            "Session started",
            extra={"user_id": user_id[:8], "initial_input_length": len(initial_input)}
        )

        return session

    def _extract_message_text(self, response: Any) -> str:
        """
        Extract text from various response formats.

        Args:
            response: Response object from agent

        Returns:
            Extracted text string
        """
        # Handle string responses
        if isinstance(response, str):
            return response

        # Handle objects with .text attribute
        if hasattr(response, 'text'):
            return response.text

        # Handle objects with .message dict
        if hasattr(response, 'message'):
            if isinstance(response.message, dict):
                return response.message.get('text', str(response.message))
            return str(response.message)

        # Handle dict responses
        if isinstance(response, dict):
            return response.get('text', str(response))

        # Fallback to string conversion
        return str(response)

    def _calculate_workflow_duration(self, session: Dict[str, Any]) -> float:
        """
        Calculate workflow duration in milliseconds.

        Args:
            session: User session dictionary

        Returns:
            Duration in milliseconds
        """
        if "created_at" not in session:
            return 0.0

        try:
            created_at = datetime.fromisoformat(session["created_at"])
            duration = (datetime.now(UTC) - created_at).total_seconds() * 1000
            return duration
        except Exception:
            return 0.0

    async def run_education(self, session: Dict[str, Any], user_input: str) -> str:
        """
        Run education phase - answer vaccine questions.
        Args:
            session: User session
            user_input: User's question
        Returns: Agent response text
        """
        trace_ctx = get_current_trace_context()
        span_context = trace_ctx.span("run_education") if trace_ctx else contextlib.suppress()

        with span_context:
            try:
                session["history"].append({
                    "role": "user",
                    "text": user_input,
                    "timestamp": datetime.now(UTC).isoformat()
                })

                response = await self.vaccine_info.emit({"text": user_input}, session=session)
                msg = self._extract_message_text(response)
                session["workflow_state"] = WorkflowState.EDUCATION.value
                session["history"].append({
                    "role": "assistant",
                    "text": msg,
                    "timestamp": datetime.now(UTC).isoformat()
                })

                metrics.counter("education_queries")
                metrics.counter("education_queries", labels={"status": "success"})

                logger.info(
                    "Education query completed",
                    extra={
                        "user_id": session.get("user_id", "unknown")[:8],
                        "response_length": len(msg)
                    }
                )
                return msg
            except Exception as e:
                metrics.counter("education_queries", labels={"status": "error"})
                logger.error("Education query failed", extra={"error": str(e)})
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
                    extra={
                        "user_id": user_id[:8],
                        "clinic_id": confirmation.get("clinic_id")
                    }
                )

                return confirmation

            except ClinicSearchError as e:
                logger.error("Clinic search failed", extra={"error": str(e)})
                session["workflow_state"] = WorkflowState.FAILED.value
                return {"confirmed": False, "reason": "clinic_search_error", "error": str(e)}

            except AppointmentBookingError as e:
                logger.error("Booking failed", extra={"error": str(e)})
                session["workflow_state"] = WorkflowState.FAILED.value
                return {"confirmed": False, "reason": "booking_error", "error": str(e)}

            except Exception as e:
                logger.exception("Unexpected workflow error", extra={"error": str(e)})
                session["workflow_state"] = WorkflowState.FAILED.value
                metrics.counter("workflow_failures", labels={"stage": "unknown"})
                return {"confirmed": False, "reason": "unexpected_error", "error": str(e)}

    async def _find_clinics(self, session: Dict[str, Any], location_query: str) -> list:
        """Find clinics with circuit breaker protection."""
        if not self.clinic_search_breaker.can_execute():
            raise ClinicSearchError("Clinic search service temporarily unavailable")

        session["location_query"] = location_query
        session["workflow_state"] = WorkflowState.CLINIC_SEARCH.value

        logger.info("Starting clinic search", extra={"location_query": location_query})

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

            logger.info("Clinic search completed", extra={"candidates_found": len(candidates)})

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

        logger.info("Starting appointment booking", extra={"candidate_count": len(candidates)})

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
                    logger.info("Confirmation saved", extra={"filename": filepath})
                except Exception as e:
                    logger.warning("Failed to save confirmation", extra={"error": str(e)})

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

        logger.info(
            "Scheduling follow-up",
            extra={"user_id": session.get("user_id", "unknown")[:8]}
        )

        try:
            await self.followup_agent.emit({}, session=session)
            metrics.counter("followups_scheduled")
            logger.info("Follow-up scheduled")
        except Exception as e:
            logger.warning("Follow-up scheduling failed (non-critical)", extra={"error": str(e)})
            metrics.counter("followup_failures")

    async def _record_analytics(self, session: Dict[str, Any], confirmation: Dict[str, Any]):
        """Record anonymized analytics."""
        anon_record = {
            "event": "appointment_confirmed",
            "clinic_id": confirmation.get("clinic_id"),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        try:
            await self.analytics_agent.emit(
                {"action": "ingest", "record": anon_record},
                session=None
            )
            metrics.counter("analytics_records_ingested")
        except Exception as e:
            logger.warning("Analytics recording failed (non-critical)", extra={"error": str(e)})

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
                extra={
                    "trace_id": trace.trace_id,
                    "total_duration_ms": summary["total_duration_ms"],
                    "confirmed": confirmation.get("confirmed")
                }
            )

            return {
                "session": session,
                "confirmation": confirmation,
                "trace": summary
            }

    async def stream_query(
            self,
            session_id: str,
            message: str,
            user_id: str,
            **kwargs,
    ) -> EventActions:
        """
        The required public entry point for the ADK runtime.
        Starts a session and executes the first phase of the workflow.
        """
        # 1. Start or retrieve the session
        session = self.start_session(user_id, message)

        # 2. Run the first phase of the workflow (Education)
        # Note: We are simplifying the flow here for the initial run
        await self.run_education(session, message)

        # 3. Return a final event to the user
        last_response = session['history'][-1]['text']

        return EventActions(
            response=last_response,
            state_delta={
                "current_state": session['workflow_state'],
                "history_length": len(session['history'])
            }
        )

    @property
    def model(self) -> str:
        """Expose the configured model name for ADK metadata."""
        return self.config.get("model", "orchestrator")

    @property
    def tools(self) -> list:
        """
        Required property for ADK Web UI discovery.
        Returns an empty list as the orchestrator itself doesn't expose tools.
        """
        return []

    @property
    def config(self):
        return self._config

    @property
    def session_service(self):
        return self._session_service

    @property
    def memory_bank(self):
        return self._memory_bank

    @property
    def vaccine_info(self):
        return self._vaccine_info

    @vaccine_info.setter
    def vaccine_info(self, value):
        """Allow setting vaccine_info for testing."""
        object.__setattr__(self, '_vaccine_info', value)

    @property
    def clinic_finder(self):
        return self._clinic_finder

    @clinic_finder.setter
    def clinic_finder(self, value):
        """Allow setting clinic_finder for testing."""
        object.__setattr__(self, '_clinic_finder', value)

    @property
    def appointment_agent(self):
        return self._appointment_agent

    @appointment_agent.setter
    def appointment_agent(self, value):
        """Allow setting appointment_agent for testing."""
        object.__setattr__(self, '_appointment_agent', value)

    @property
    def followup_agent(self):
        return self._followup_agent

    @followup_agent.setter
    def followup_agent(self, value):
        """Allow setting followup_agent for testing."""
        object.__setattr__(self, '_followup_agent', value)

    @property
    def analytics_agent(self):
        return self._analytics_agent

    @analytics_agent.setter
    def analytics_agent(self, value):
        """Allow setting analytics_agent for testing."""
        object.__setattr__(self, '_analytics_agent', value)

    @property
    def clinic_search_breaker(self):
        return self._clinic_search_breaker

    @property
    def booking_breaker(self):
        return self._booking_breaker

    @property
    def secure_storage(self):
        return self._secure_storage