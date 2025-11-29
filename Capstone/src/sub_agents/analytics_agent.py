"""
Analytics Agent

Responsibilities:
- Ingest anonymized events (appointment confirmations, follow-up reports).
- Maintain lightweight in-memory metrics and counters.
- Optionally push aggregates to a persistent store (placeholder hooks).
- Expose simple APIs for the orchestrator to request aggregates.

Features:
- Demonstrates Observability (logging/metrics).
- Demonstrates MemoryBank integration (reads anonymized records).
- Keeps privacy in mind: expects anonymized records (no PII).
"""

import logging
import threading
from typing import Dict, Any, List, Optional
from collections import Counter
from datetime import datetime
from pydantic import PrivateAttr
from pydantic import BaseModel, Field
from datetime import UTC

# Try to import ADK Agent; fallback if not available for tests
try:
    from google.adk import Agent
    from google.adk.events import EventActions
except ImportError:
    # Define simple mock classes for testing outside the full ADK environment
    class Agent:
        # Agent class uses __init__ with various arguments (name, model, etc.)
        def __init__(self, *args, **kwargs):
            pass

    class EventActions:
        # EventActions is called to return a flow control action (restart, pause, etc.)
        def __init__(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MetricsCollector:
    """Simple in-memory metrics collector. Replace with Prometheus/OpenTelemetry in prod."""

    def __init__(self):
        self._lock = threading.Lock()
        self.counters = Counter()
        self.gauges = {}

    def increment(self, name: str, amount: int = 1):
        with self._lock:
            self.counters[name] += amount

    def set_gauge(self, name: str, value: Any):
        with self._lock:
            self.gauges[name] = value

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {"counters": dict(self.counters), "gauges": dict(self.gauges)}

class AnalyticsAgent(Agent):

    # Declare private attributes using PrivateAttr
    _memory_bank: Any = PrivateAttr(default=None)
    _config: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _records: List[Dict[str, Any]] = PrivateAttr(default_factory=dict)

    def __init__(self, config: Dict[str, Any], memory_bank=None):
        super().__init__(
            name="analytics_agent",
            model=config.get("model"),
            description="Aggregates anonymized feedback and appointment metrics for reporting."
                        "Ingest anonymized records and provide aggregated metrics on request.",
        )
        # Use private attributes to avoid Pydantic field conflicts
        object.__setattr__(self, '_config', config)
        object.__setattr__(self, '_memory_bank', memory_bank)
        object.__setattr__(self, '_lock', threading.Lock())
        object.__setattr__(self, '_records', [])
        object.__setattr__(self, '_metrics', MetricsCollector())

    @property
    def config(self):
        return self._config

    @property
    def memory_bank(self):
        return self._memory_bank

    @property
    def metrics(self) -> 'MetricsCollector':
        return self._metrics

    @property
    def records(self) -> List[Dict[str, Any]]:
        return self._records

    async def on_event(self, event, ctx):
        """
        event.payload can be:
         - {"action": "ingest", "record": {...}} to ingest a new anonymized record
         - {"action": "aggregate"} to return aggregated metrics
         - {"action": "export"} to trigger a placeholder export job
        """
        payload = event.payload or {}
        action = payload.get("action", "aggregate")
        logger.info("AnalyticsAgent received action=%s", action)

        if action == "ingest":
            record = payload.get("record")
            if record:
                self._ingest(record)
                # increment metric
                self.metrics.increment("records_ingested")
                return EventActions(state_delta={"status": "ingested"})
            else:
                return EventActions(state_delta={"error": "no record provided"})

        if action == "aggregate":
            agg = self._aggregate()
            return EventActions(state_delta={"aggregate": agg})

        if action == "export":
            # placeholder for exporting to BigQuery / dashboard
            exported = self._export_placeholder()
            return EventActions(state_delta={"exported": exported})

        return EventActions(state_delta={"error": "unknown action"})

    async def _mock_call_tool(self, tool_name: str, params: dict):
        """Mock tool call (not used by this agent)."""
        return {}

    async def emit(self, payload: dict, session: dict):
        """Convenience method for testing and direct invocation."""
        fake_event = type("Event", (), {"payload": payload, "resume": False})()
        fake_ctx = type("Context", (), {
            "session": session,
            "metrics": None,
            "call_tool": self._mock_call_tool  # Keep for compatibility
        })()

        response = await self.on_event(fake_event, fake_ctx)
        return response

    def _ingest(self, record: Dict[str, Any]):
        """Store an anonymized record. Production systems should apply DP/k-anonymity here."""
        logger.info("AnalyticsAgent ingested record: keys=%s", list(record.keys()))
        # Validate record shape
        r = record.copy()
        r["_received_at"] = datetime.now(UTC).isoformat()
        if "timestamp" not in record:
            r["timestamp"] = datetime.now(UTC).isoformat()

        self._records.append(r)

        # Update metrics based on the event type
        event_type = r.get("event", "unknown")
        self.metrics.increment(f"ingest_event:{event_type}", 1)

    def _aggregate(self) -> Dict[str, Any]:
        """Return simple aggregates computed over ingested records."""
        total = len(self._records)
        appointment_confirmed = sum(1 for r in self._records if r.get("event") == "appointment_confirmed")
        followup_reports = sum(1 for r in self._records if r.get("event") == "followup_report")
        symptom_reports = sum(1 for r in self._records if r.get("reported_symptoms"))
        # attach metric snapshot
        metrics_snapshot = self.metrics.snapshot()
        return {
            "total_records": total,
            "appointment_confirmed": appointment_confirmed,
            "followup_reports": followup_reports,
            "symptom_reports": symptom_reports,
            "metrics": metrics_snapshot,
        }

    def _export_placeholder(self) -> Dict[str, Any]:
        """Placeholder export - replace with BigQuery/Cloud Storage export in production."""
        exported_at = datetime.now(UTC).isoformat()
        logger.info("Analytics export placeholder invoked at %s", exported_at)
        return {"exported_at": exported_at, "records_exported": len(self._records)}

    # Helper used by orchestrator to push MemoryBank entries into analytics
    def ingest_from_memorybank(self):
        if not self.memory_bank:
            return
        # Expect memory bank to return anonymized entries or we sanitize here
        for user_id, items in list(self.memory_bank._store.items()):
            for it in items:
                # Sanitize — remove user_id and any PII
                record = it.copy()
                record.pop("user_id", None)
                record.pop("email", None)
                self._ingest(record)
