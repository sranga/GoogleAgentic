"""
VAccess Orchestrator

This orchestrator:
- is async to await agent emits
- integrates ClinicFinderAgent, AppointmentAgent, FollowUpAgent, AnalyticsAgent
- uses InMemorySessionService and MemoryBank
- demonstrates sequential orchestration + parallel prefetching (via ClinicFinder)
- logs important events and pushes anonymized records to analytics
"""

import asyncio
import logging
from typing import Dict, Any

from sub_agents.vaccine_info_agent import VaccineInfoAgent
from sub_agents.clinic_finder_agent import ClinicFinderAgent
from sub_agents.appointment_agent import AppointmentAgent
from sub_agents.followup_agent import FollowUpAgent
from sub_agents.analytics_agent import AnalyticsAgent
from memory import InMemorySessionService, MemoryBank
from tools import save_confirmation_to_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VAccessOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        # session & memory
        self.session_service = InMemorySessionService()
        self.memory_bank = MemoryBank()
        # instantiate agents (these expect ADK-style Agent classes)
        self.vaccine_info = VaccineInfoAgent(config)
        self.clinic_finder = ClinicFinderAgent(config)
        self.appointment_agent = AppointmentAgent(config)
        self.followup_agent = FollowUpAgent(config, memory_bank=self.memory_bank)
        self.analytics_agent = AnalyticsAgent(config, memory_bank=self.memory_bank)
        # simple in-process event loop semaphore
        self._lock = asyncio.Lock()

    def start_session(self, user_id: str, initial_input: str):
        session = self.session_service.create_session(user_id)
        session["history"].append({"role": "user", "text": initial_input})
        return session

    async def run_education(self, session, user_input: str):
        """Synchronous-style education: call vaccine_info agent and return a string."""
        # Assuming VaccineInfoAgent exposes an async emit() or on_event wrapper
        try:
            response = await self.vaccine_info.emit({"text": user_input}, session=session)
            # Expect response to be a ModelMessage or EventActions with message
            msg = getattr(response, "message", None) or getattr(response, "text", None) or response
            if isinstance(msg, dict) and "text" in msg:
                msg = msg["text"]
            session["history"].append({"role": "assistant", "text": str(msg)})
            return str(msg)
        except Exception as e:
            logger.exception("Education run failed: %s", e)
            return "Sorry, I couldn't retrieve vaccine info at the moment."

    async def find_and_schedule(self, session, location_query: str):
        """Full end-to-end find + schedule + follow-up orchestration (async)."""
        async with self._lock:
            # 1) find clinics
            session["location_query"] = location_query
            logger.info("Orchestrator: invoking clinic finder for query=%s", location_query)
            clinic_resp = await self.clinic_finder.emit({"location_query": location_query}, session=session)
            # clinic_resp.message may contain candidates or EventActions with message
            candidates = None
            try:
                candidates = (clinic_resp.message or {}).get("candidates") if hasattr(clinic_resp, "message") else None
            except Exception:
                candidates = None

            # Fallback: consult session
            candidates = candidates or session.get("last_clinics") or []

            if not candidates:
                logger.warning("No clinics found for query=%s", location_query)
                return {"confirmed": False, "reason": "no_clinics_found"}

            # 2) invoke appointment agent (sequential)
            session["last_clinics"] = candidates
            logger.info("Orchestrator: invoking appointment agent with %d candidates", len(candidates))
            try:
                appointment_result = await self.appointment_agent.emit({}, session=session)
                # Tool result or EventActions with a message
                # Normalize appointment_result to dict
                result_obj = None
                if hasattr(appointment_result, "message") and isinstance(appointment_result.message, dict):
                    result_obj = appointment_result.message
                else:
                    result_obj = appointment_result
                # If the appointment agent called a tool and returned tool output, handle that
                if isinstance(result_obj, dict) and result_obj.get("confirmed"):
                    confirmation = result_obj
                    # Save confirmation to file (local artifact)
                    try:
                        fname = save_confirmation_to_file(session["user_id"], confirmation)
                        logger.info("Saved confirmation artifact: %s", fname)
                    except Exception as e:
                        logger.exception("Failed to save confirmation file: %s", e)
                    # store anonymized analytics record
                    anon_record = {"event": "appointment_confirmed", "clinic_id": confirmation.get("slot", {}).get("clinic_id")}
                    try:
                        await self.analytics_agent.emit({"action": "ingest", "record": anon_record}, session=None)
                    except Exception:
                        # fallback to direct ingest
                        self.analytics_agent._ingest(anon_record)

                    # 3) schedule follow-up (hand off to follow-up agent which will pause)
                    logger.info("Scheduling follow-up via FollowUpAgent")
                    try:
                        await self.followup_agent.emit({}, session=session)
                    except Exception as e:
                        logger.exception("Follow-up scheduling failed: %s", e)

                    return confirmation
                else:
                    logger.warning("Appointment agent returned no confirmation: %s", result_obj)
                    return {"confirmed": False, "reason": "no_confirmation"}
            except Exception as e:
                logger.exception("Appointment flow failed: %s", e)
                return {"confirmed": False, "reason": "exception"}

    async def run_followup_analytics(self):
        """Example utility that pulls memorybank entries into analytics."""
        try:
            # Let analytics agent ingest sanitized items from memory bank
            self.analytics_agent.ingest_from_memorybank()
            agg = self.analytics_agent._aggregate()
            logger.info("Analytics aggregate: %s", agg)
            return agg
        except Exception as e:
            logger.exception("run_followup_analytics failed: %s", e)
            return {}

    # Convenience wrapper to run a full scenario for quick demos
    async def run_demo_flow(self, user_id: str, location_query: str):
        session = self.start_session(user_id, "Hi, I need a vaccine")
        # 1) education
        await self.run_education(session, "What vaccines are available?")
        # 2) find + schedule
        confirmation = await self.find_and_schedule(session, location_query)
        # 3) return overall summary
        return {"session": session, "confirmation": confirmation}
