"""Small dependency-free request metrics and privacy-conscious audit events."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import logging
import threading
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

_lock = threading.Lock()
_requests = Counter()
_durations = defaultdict(list)
_stage_counts = Counter()
_stage_durations = defaultdict(list)
_audit = logging.getLogger("securerag.audit")
_audit.setLevel(logging.INFO)
_audit.propagate = False
if not _audit.handlers:
    _audit.addHandler(logging.StreamHandler())


def audit_event(event: str, **fields) -> None:
    """Log event metadata only; callers must not pass secrets or document text."""
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    _audit.info(json.dumps(record, separators=(",", ":"), default=str))


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            duration = time.perf_counter() - started
            route = request.scope.get("route")
            path = getattr(route, "path", "/unmatched")
            with _lock:
                _requests[(request.method, path, status)] += 1
                values = _durations[(request.method, path)]
                values.append(duration)
                del values[:-10000]


def prometheus_text() -> str:
    lines = ["# HELP securerag_http_requests_total HTTP requests by method, route and status.",
             "# TYPE securerag_http_requests_total counter"]
    with _lock:
        for (method, path, status), count in sorted(_requests.items()):
            labels = f'method="{method}",route="{path}",status="{status}"'
            lines.append(f"securerag_http_requests_total{{{labels}}} {count}")
        lines += ["# HELP securerag_http_request_duration_seconds HTTP request duration.",
                  "# TYPE securerag_http_request_duration_seconds summary"]
        for (method, path), values in sorted(_durations.items()):
            labels = f'method="{method}",route="{path}"'
            lines.append(f"securerag_http_request_duration_seconds_count{{{labels}}} {len(values)}")
            lines.append(f"securerag_http_request_duration_seconds_sum{{{labels}}} {sum(values):.6f}")
        lines += ["# HELP securerag_pipeline_stage_total Completed retrieval and ingestion stages.",
                  "# TYPE securerag_pipeline_stage_total counter"]
        for (stage, outcome), count in sorted(_stage_counts.items()):
            lines.append(f'securerag_pipeline_stage_total{{stage="{stage}",outcome="{outcome}"}} {count}')
        lines += ["# HELP securerag_pipeline_stage_duration_seconds_sum Pipeline stage duration.",
                  "# TYPE securerag_pipeline_stage_duration_seconds summary"]
        for (stage, outcome), values in sorted(_stage_durations.items()):
            labels = f'stage="{stage}",outcome="{outcome}"'
            lines.append(f"securerag_pipeline_stage_duration_seconds_count{{{labels}}} {len(values)}")
            lines.append(f"securerag_pipeline_stage_duration_seconds_sum{{{labels}}} {sum(values):.6f}")
    return "\n".join(lines) + "\n"


def record_stage(stage: str, duration: float, outcome: str = "success") -> None:
    with _lock:
        key = (stage, outcome)
        _stage_counts[key] += 1
        values = _stage_durations[key]
        values.append(max(0.0, duration))
        del values[:-10000]


def stage_snapshot() -> dict:
    with _lock:
        return {key: (len(values), sum(values)) for key, values in _stage_durations.items()}


def stage_delta(before: dict, after: dict) -> dict:
    result = {}
    for key, (count, total) in after.items():
        old_count, old_total = before.get(key, (0, 0.0))
        added_count = count - old_count
        added_total = total - old_total
        if added_count > 0:
            result[f"{key[0]}:{key[1]}"] = {
                "count": added_count,
                "mean_ms": round(added_total * 1000 / added_count, 3),
            }
    return result
