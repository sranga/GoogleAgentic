"""
Observability Module

Production-grade observability for the V-Access system including:
- Structured logging with trace context
- Metrics collection (counters, histograms, gauges)
- Distributed tracing
- Health checks
"""

import logging
import time
import threading
import json
import uuid
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from datetime import datetime, UTC
from collections import defaultdict


# ============================================================================
# STRUCTURED LOGGING
# ============================================================================

class StructuredLogFormatter(logging.Formatter):
    """JSON formatter for cloud logging compatibility."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        if hasattr(record, "trace_id"):
            log_data["trace_id"] = record.trace_id
        if hasattr(record, "span_id"):
            log_data["span_id"] = record.span_id

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "pathname", "process", "processName",
                "relativeCreated", "thread", "threadName", "exc_info",
                "exc_text", "stack_info", "trace_id", "span_id"
            ]:
                log_data[key] = value

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class TraceContextFilter(logging.Filter):
    """Adds trace context to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        trace_ctx = get_current_trace_context()
        if trace_ctx:
            record.trace_id = trace_ctx.trace_id
            record.span_id = trace_ctx.current_span_id
        return True


def get_logger(name: str, use_json: bool = True) -> logging.Logger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)
        use_json: Use JSON formatting (recommended for production)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()

        if use_json:
            handler.setFormatter(StructuredLogFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))

        handler.addFilter(TraceContextFilter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


# ============================================================================
# METRICS
# ============================================================================

class ProductionMetrics:
    """
    Thread-safe metrics collector with counters, gauges, and histograms.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._histogram_max_size = 10000

    def counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] += value

    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge value."""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram value."""
        key = self._make_key(name, labels)
        with self._lock:
            hist = self._histograms[key]
            hist.append(value)
            if len(hist) > self._histogram_max_size:
                self._histograms[key] = hist[-self._histogram_max_size:]

    def get_percentile(self, name: str, percentile: float,
                       labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Calculate percentile from histogram."""
        key = self._make_key(name, labels)
        with self._lock:
            values = self._histograms.get(key, [])
            if not values:
                return None
            sorted_values = sorted(values)
            index = int(len(sorted_values) * percentile / 100)
            return sorted_values[min(index, len(sorted_values) - 1)]

    def snapshot(self) -> Dict[str, Any]:
        """Get snapshot of all metrics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histogram_counts": {k: len(v) for k, v in self._histograms.items()},
                "timestamp": datetime.now(UTC).isoformat()
            }

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# Global metrics instance
metrics = ProductionMetrics()


# ============================================================================
# DISTRIBUTED TRACING
# ============================================================================

_trace_context_storage = threading.local()


class TraceContext:
    """Distributed tracing context for tracking requests across agents."""

    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.spans: List[Dict[str, Any]] = []
        self.current_span_id: Optional[str] = None
        self._span_stack: List[str] = []

    @contextmanager
    def span(self, name: str, **attributes):
        """Create a new span."""
        span_id = str(uuid.uuid4())
        parent_span_id = self.current_span_id
        start_time = time.time()

        old_span_id = self.current_span_id
        self.current_span_id = span_id
        self._span_stack.append(span_id)

        span_data = {
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "name": name,
            "start_time": start_time,
            "attributes": attributes,
        }

        try:
            yield span_id
            span_data["status"] = "success"
        except Exception as e:
            span_data["status"] = "error"
            span_data["error"] = str(e)
            raise
        finally:
            span_data["end_time"] = time.time()
            span_data["duration_ms"] = (span_data["end_time"] - start_time) * 1000
            self.spans.append(span_data)
            self._span_stack.pop()
            self.current_span_id = old_span_id

    def get_trace_summary(self) -> Dict[str, Any]:
        """Get summary of the trace."""
        if not self.spans:
            return {"trace_id": self.trace_id, "spans": [], "total_duration_ms": 0}

        total_duration = sum(s.get("duration_ms", 0) for s in self.spans)

        return {
            "trace_id": self.trace_id,
            "total_duration_ms": total_duration,
            "span_count": len(self.spans),
            "spans": self.spans,
        }


def set_trace_context(trace: TraceContext):
    """Set trace context for current thread."""
    _trace_context_storage.trace = trace


def get_current_trace_context() -> Optional[TraceContext]:
    """Get trace context for current thread."""
    return getattr(_trace_context_storage, "trace", None)


@contextmanager
def trace_request(operation: str = "request"):
    """Convenience context manager for tracing a request."""
    trace = TraceContext()
    set_trace_context(trace)

    try:
        with trace.span(operation):
            yield trace
    finally:
        logger = get_logger(__name__)
        summary = trace.get_trace_summary()
        logger.info(
            "Request completed",
            trace_id=trace.trace_id,
            total_duration_ms=summary["total_duration_ms"],
            span_count=summary["span_count"]
        )


# ============================================================================
# HEALTH CHECKS
# ============================================================================

class HealthChecker:
    """System health checker."""

    def __init__(self):
        self._checks: Dict[str, callable] = {}

    def register_check(self, name: str, check_fn: callable):
        """Register a health check function."""
        self._checks[name] = check_fn

    async def check_health(self) -> Dict[str, Any]:
        """Run all health checks."""
        import asyncio

        results = {}
        all_healthy = True

        for name, check_fn in self._checks.items():
            try:
                if asyncio.iscoroutinefunction(check_fn):
                    is_healthy = await check_fn()
                else:
                    is_healthy = check_fn()
                results[name] = {"status": "healthy" if is_healthy else "unhealthy"}
                if not is_healthy:
                    all_healthy = False
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
                all_healthy = False

        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "checks": results,
            "timestamp": datetime.now(UTC).isoformat()
        }


# Global health checker
health_checker = HealthChecker()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "get_logger",
    "metrics",
    "TraceContext",
    "set_trace_context",
    "get_current_trace_context",
    "trace_request",
    "health_checker",
]