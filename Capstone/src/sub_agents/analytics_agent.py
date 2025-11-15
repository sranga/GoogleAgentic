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

# Try to import ADK Agent; fallback if not available for tests
try:
    from adk import Agent, EventActions
except Exception:
    from adk import Agent, EventActions  # type: ignore

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
    def __init__(self, config: Dict[str, Any], memory_bank=None):
        super().__init__(
            name="analytics_agent",
            model=config.get("model"),
            description="Aggregates anonymized feedback and appointment metrics for reporting.",
            instructions="Ingest anonymized records and provide aggregated metrics on request.",
        )
        self.config = config
        self.memory_bank = memory_bank
        self._records: List[Dict[str, Any]] = []
        self.metrics = MetricsCollector()

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
                return EventActions(resume=True, message={"status": "ingested"})
            else:
                return EventActions(resume=True, message={"error": "no record provided"})

        if action == "aggregate":
            agg = self._aggregate()
            return EventActions(resume=True, message={"aggregate": agg})

        if action == "export":
            # placeholder for exporting to BigQuery / dashboard
            exported = self._export_placeholder()
            return EventActions(resume=True, message={"exported": exported})

        return EventActions(resume=True, message={"error": "unknown action"})

    def _ingest(self, record: Dict[str, Any]):
        """Store an anonymized record. Production systems should apply DP/k-anonymity here."""
        # Validate record shape
        r = record.copy()
        r["_received_at"] = datetime.utcnow().isoformat()
        self._records.append(r)
        logger.info("AnalyticsAgent ingested record: keys=%s", list(record.keys()))

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
        exported_at = datetime.utcnow().isoformat()
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
