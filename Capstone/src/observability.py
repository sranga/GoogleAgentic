"""
Production-grade observability module for V-Access.

Provides:
- Structured logging with trace context
- Metrics collection (counters, histograms, gauges)
- Distributed tracing
- Performance monitoring
- Error tracking with context

Usage:
    from observability import get_logger, metrics, TraceContext

    logger = get_logger(__name__)
    logger.info("Event occurred", user_id="123", action="search")

    metrics.counter("api_calls", labels={"endpoint": "/search", "status": "success"})
    metrics.histogram("request_duration_ms", 45.2, labels={"endpoint": "/search"})
"""

import logging
import time
import threading
import json
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from datetime import datetime
from collections import defaultdict
import uuid


# ============================================================================
# STRUCTURED LOGGING
# ============================================================================

class StructuredLogFormatter(logging.Formatter):
    """
    Formats logs as JSON with trace context and structured fields.
    Compatible with cloud logging systems (CloudWatch, Stackdriver, etc.)
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add trace context if present
        if hasattr(record, "trace_id"):
            log_data["trace_id"] = record.trace_id
        if hasattr(record, "span_id"):
            log_data["span_id"] = record.span_id

        # Add custom fields from extra parameter
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "created", "filename", "funcName",
                           "levelname", "levelno", "lineno", "module", "msecs",
                           "message", "pathname", "process", "processName",
                           "relativeCreated", "thread", "threadName", "exc_info",
                           "exc_text", "stack_info", "trace_id", "span_id"]:
                log_data[key] = value

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class TraceContextFilter(logging.Filter):
    """
    Adds trace context to all log records if available in thread-local storage.
    """

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
        use_json: Whether to use JSON formatting (recommended for production)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()

        if use_json:
            formatter = StructuredLogFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        handler.setFormatter(formatter)
        handler.addFilter(TraceContextFilter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


# ============================================================================
# METRICS COLLECTION
# ============================================================================

class ProductionMetrics:
    """
    Production-grade metrics collector with support for:
    - Counters (monotonically increasing values)
    - Gauges (point-in-time values)
    - Histograms (distribution of values with percentiles)
    - Labels/dimensions for metric segmentation

    Thread-safe for concurrent access.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._histogram_max_size = 10000  # Prevent unbounded growth

    def counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """
        Increment a counter metric.

        Args:
            name: Metric name (e.g., "api_requests_total")
            value: Amount to increment (default: 1)
            labels: Optional labels for metric segmentation

        Example:
            metrics.counter("clinic_searches", labels={"method": "google_maps", "status": "success"})
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] += value

    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Set a gauge metric (point-in-time value).

        Args:
            name: Metric name (e.g., "active_sessions")
            value: Current value
            labels: Optional labels

        Example:
            metrics.gauge("memory_usage_mb", 512.3, labels={"agent": "clinic_finder"})
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Record a histogram value (for latency, size distributions, etc.).

        Args:
            name: Metric name (e.g., "request_duration_ms")
            value: Observed value
            labels: Optional labels

        Example:
            metrics.histogram("agent_duration_ms", 45.2, labels={"agent": "vaccine_info"})
        """
        key = self._make_key(name, labels)
        with self._lock:
            hist = self._histograms[key]
            hist.append(value)

            # Prevent unbounded growth - keep only recent values
            if len(hist) > self._histogram_max_size:
                self._histograms[key] = hist[-self._histogram_max_size:]

    def get_percentile(self, name: str, percentile: float,
                       labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """
        Calculate percentile from histogram data.

        Args:
            name: Metric name
            percentile: Percentile to calculate (0-100, e.g., 95 for p95)
            labels: Optional labels

        Returns:
            Percentile value or None if no data
        """
        key = self._make_key(name, labels)
        with self._lock:
            values = self._histograms.get(key, [])
            if not values:
                return None

            sorted_values = sorted(values)
            index = int(len(sorted_values) * percentile / 100)
            return sorted_values[min(index, len(sorted_values) - 1)]

    def get_histogram_stats(self, name: str,
                            labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """
        Get summary statistics for a histogram.

        Returns:
            Dict with min, max, mean, p50, p95, p99, count
        """
        key = self._make_key(name, labels)
        with self._lock:
            values = self._histograms.get(key, [])
            if not values:
                return {}

            sorted_values = sorted(values)
            count = len(sorted_values)

            return {
                "count": count,
                "min": sorted_values[0],
                "max": sorted_values[-1],
                "mean": sum(sorted_values) / count,
                "p50": sorted_values[int(count * 0.50)],
                "p95": sorted_values[int(count * 0.95)],
                "p99": sorted_values[int(count * 0.99)],
            }

    def snapshot(self) -> Dict[str, Any]:
        """
        Get a complete snapshot of all metrics.
        Useful for exporting to monitoring systems.
        """
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    key: self.get_histogram_stats(key.split("{")[0], self._parse_labels(key))
                    for key in self._histograms.keys()
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

    def reset(self):
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key from metric name and labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _parse_labels(self, key: str) -> Optional[Dict[str, str]]:
        """Parse labels from a metric key."""
        if "{" not in key:
            return None

        label_str = key.split("{")[1].rstrip("}")
        if not label_str:
            return None

        labels = {}
        for pair in label_str.split(","):
            k, v = pair.split("=")
            labels[k] = v
        return labels


# Global metrics instance
metrics = ProductionMetrics()

# ============================================================================
# DISTRIBUTED TRACING
# ============================================================================

_trace_context_storage = threading.local()


class TraceContext:
    """
    Distributed tracing context for tracking requests across agents.

    Usage:
        trace = TraceContext()
        with trace.span("clinic_search", agent="clinic_finder"):
            # ... do work ...
            with trace.span("api_call", operation="google_maps"):
                # ... nested span ...
    """

    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.spans: List[Dict[str, Any]] = []
        self.current_span_id: Optional[str] = None
        self._span_stack: List[str] = []

    @contextmanager
    def span(self, name: str, **attributes):
        """
        Create a new span for tracing an operation.

        Args:
            name: Span name (e.g., "clinic_search", "llm_call")
            **attributes: Additional span attributes (agent, operation, etc.)

        Example:
            with trace.span("appointment_booking", agent="appointment_agent", clinic_id="123"):
                result = await book_appointment()
        """
        span_id = str(uuid.uuid4())
        parent_span_id = self.current_span_id
        start_time = time.time()

        # Set as current span
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
            end_time = time.time()
            span_data["end_time"] = end_time
            span_data["duration_ms"] = (end_time - start_time) * 1000

            self.spans.append(span_data)
            self._span_stack.pop()
            self.current_span_id = old_span_id

    def get_trace_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire trace."""
        if not self.spans:
            return {"trace_id": self.trace_id, "spans": []}

        total_duration = sum(s["duration_ms"] for s in self.spans)

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
    """
    Convenience context manager to create and track a new trace.

    Usage:
        with trace_request("user_vaccine_search"):
            # ... handle request ...
    """
    trace = TraceContext()
    set_trace_context(trace)

    try:
        with trace.span(operation):
            yield trace
    finally:
        # Log trace summary
        logger = get_logger(__name__)
        summary = trace.get_trace_summary()
        logger.info(
            "Request completed",
            trace_id=trace.trace_id,
            total_duration_ms=summary["total_duration_ms"],
            span_count=summary["span_count"]
        )


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

@contextmanager
def monitor_performance(operation: str, labels: Optional[Dict[str, str]] = None):
    """
    Context manager to automatically track operation duration.

    Usage:
        with monitor_performance("clinic_search", labels={"method": "google_maps"}):
            results = search_clinics()
    """
    start_time = time.time()
    error_occurred = False

    try:
        yield
    except Exception:
        error_occurred = True
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000

        # Record metrics
        metric_labels = labels or {}
        metric_labels["status"] = "error" if error_occurred else "success"

        metrics.histogram(f"{operation}_duration_ms", duration_ms, labels=metric_labels)
        metrics.counter(f"{operation}_total", labels=metric_labels)


# ============================================================================
# HEALTH CHECK
# ============================================================================

class HealthChecker:
    """
    System health checker for monitoring agent and service health.
    """

    def __init__(self):
        self._checks: Dict[str, callable] = {}

    def register_check(self, name: str, check_fn: callable):
        """Register a health check function."""
        self._checks[name] = check_fn

    async def check_health(self) -> Dict[str, Any]:
        """
        Run all health checks and return status.

        Returns:
            Dict with overall status and individual check results
        """
        results = {}
        all_healthy = True

        for name, check_fn in self._checks.items():
            try:
                is_healthy = await check_fn() if asyncio.iscoroutinefunction(check_fn) else check_fn()
                results[name] = {"status": "healthy" if is_healthy else "unhealthy"}
                if not is_healthy:
                    all_healthy = False
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
                all_healthy = False

        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "checks": results,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


# Global health checker
health_checker = HealthChecker()

# Export convenience imports
__all__ = [
    "get_logger",
    "metrics",
    "TraceContext",
    "set_trace_context",
    "get_current_trace_context",
    "trace_request",
    "monitor_performance",
    "health_checker",
]