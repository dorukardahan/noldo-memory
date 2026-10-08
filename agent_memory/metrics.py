"""In-process metrics collector with Prometheus text exposition."""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from typing import Any, Dict, Tuple


class MetricsCollector:
    """Thread-safe metrics collector for API/recall instrumentation."""

    STAGES = frozenset({"queue", "normalize", "dedup", "persist", "embedding", "graph",
                        "bm25", "vector", "rerank", "access", "total"})
    OPERATIONS = frozenset({"capture", "store", "recall", "status", "backend", "index"})
    OUTCOMES = frozenset({"accepted", "completed", "failed", "blocked", "timeout",
                          "queue_full", "degraded", "cancelled", "rejected"})
    JOB_STAGES = frozenset({"embed", "graph"})

    REQUEST_DURATION_BUCKETS = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
        2.5,
        5.0,
        6.0,
        8.0,
        10.0,
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Counters
        self._requests_total: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self._cache_hits_total: int = 0
        self._cache_misses_total: int = 0

        # Histogram (cumulative bucket counts)
        self._request_duration_bucket_counts: Dict[str, list[int]] = {}
        self._request_duration_sum: Dict[str, float] = defaultdict(float)
        self._request_duration_count: Dict[str, int] = defaultdict(int)

        # Gauges
        self._memories_total_by_agent: Dict[str, int] = {}
        self._vectorless_total: int = 0
        self._embed_queue_depth: int = 0
        self._stage_buckets: dict[tuple[str, str, str], list[int]] = {}
        self._stage_sum: dict[tuple[str, str, str], float] = defaultdict(float)
        self._stage_count: dict[tuple[str, str, str], int] = defaultdict(int)
        self._stage_attempts: dict[str, int] = defaultdict(int)
        self._stage_retries: dict[str, int] = defaultdict(int)
        self._job_outcomes: dict[tuple[str, str], int] = defaultdict(int)
        self._index_queue_depth = 0
        self._index_inflight = 0

    def reset(self) -> None:
        """Reset all metrics (used by tests)."""
        with self._lock:
            self._requests_total.clear()
            self._cache_hits_total = 0
            self._cache_misses_total = 0
            self._request_duration_bucket_counts.clear()
            self._request_duration_sum.clear()
            self._request_duration_count.clear()
            self._memories_total_by_agent = {}
            self._vectorless_total = 0
            self._embed_queue_depth = 0
            self._stage_buckets.clear()
            self._stage_sum.clear()
            self._stage_count.clear()
            self._stage_attempts.clear()
            self._stage_retries.clear()
            self._job_outcomes.clear()
            self._index_queue_depth = 0
            self._index_inflight = 0

    def record_stage(self, operation: str, stage: str, duration_seconds: float,
                     outcome: str = "completed") -> None:
        """Observe a finite duration using only fixed, low-cardinality labels."""
        if (not isinstance(operation, str) or operation not in self.OPERATIONS
                or not isinstance(stage, str) or stage not in self.STAGES
                or not isinstance(outcome, str) or outcome not in self.OUTCOMES):
            raise ValueError("invalid_metric_label")
        duration = _safe_duration(duration_seconds)
        key = (operation, stage, outcome)
        with self._lock:
            buckets = self._stage_buckets.setdefault(key, [0] * len(self.REQUEST_DURATION_BUCKETS))
            for i, upper in enumerate(self.REQUEST_DURATION_BUCKETS):
                if duration <= upper:
                    buckets[i] += 1
            self._stage_sum[key] += duration
            self._stage_count[key] += 1

    def record_attempt(self, stage: str, *, retry: bool = False) -> None:
        if not isinstance(stage, str) or stage not in self.STAGES or type(retry) is not bool:
            raise ValueError("invalid_metric_label")
        with self._lock:
            self._stage_attempts[stage] += 1
            if retry:
                self._stage_retries[stage] += 1

    def record_job(self, stage: str, outcome: str) -> None:
        if (not isinstance(stage, str) or stage not in self.JOB_STAGES
                or not isinstance(outcome, str) or outcome not in {"completed", "failed", "blocked"}):
            raise ValueError("invalid_metric_label")
        with self._lock:
            self._job_outcomes[(stage, outcome)] += 1

    def set_index_gauges(self, *, queue_depth: int, inflight: int) -> None:
        if type(queue_depth) is not int or type(inflight) is not int or min(queue_depth, inflight) < 0:
            raise ValueError("invalid_metric_count")
        with self._lock:
            self._index_queue_depth = queue_depth
            self._index_inflight = inflight

    def record_request(self, method: str, path: str, status: int, duration_seconds: float) -> None:
        """Record request counter and latency histogram observation."""
        method_norm = (method or "GET").upper()
        if method_norm not in {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}:
            method_norm = "OTHER"
        path_norm = normalize_metric_path(path)
        status_norm = str(status) if type(status) is int and 100 <= status <= 599 else "other"
        duration = _safe_duration(duration_seconds)

        with self._lock:
            self._requests_total[(method_norm, path_norm, status_norm)] += 1

            buckets = self._request_duration_bucket_counts.get(path_norm)
            if buckets is None:
                buckets = [0 for _ in self.REQUEST_DURATION_BUCKETS]
                self._request_duration_bucket_counts[path_norm] = buckets

            for idx, upper_bound in enumerate(self.REQUEST_DURATION_BUCKETS):
                if duration <= upper_bound:
                    buckets[idx] += 1

            self._request_duration_sum[path_norm] += duration
            self._request_duration_count[path_norm] += 1

    def inc_cache_hit(self, count: int = 1) -> None:
        with self._lock:
            self._cache_hits_total += max(0, int(count))

    def inc_cache_miss(self, count: int = 1) -> None:
        with self._lock:
            self._cache_misses_total += max(0, int(count))

    def set_runtime_gauges(
        self,
        *,
        memories_total_by_agent: Dict[str, int],
        vectorless_total: int,
        embed_queue_depth: int,
    ) -> None:
        """Update runtime gauges shown in Prometheus output."""
        cleaned = {
            str(agent): max(0, int(total))
            for agent, total in memories_total_by_agent.items()
        }
        with self._lock:
            self._memories_total_by_agent = cleaned
            self._vectorless_total = max(0, int(vectorless_total))
            self._embed_queue_depth = max(0, int(embed_queue_depth))

    def snapshot(self) -> Dict[str, Any]:
        """Take an immutable snapshot for exposition."""
        with self._lock:
            return {
                "requests_total": dict(self._requests_total),
                "request_duration_bucket_counts": {
                    path: list(counts)
                    for path, counts in self._request_duration_bucket_counts.items()
                },
                "request_duration_sum": dict(self._request_duration_sum),
                "request_duration_count": dict(self._request_duration_count),
                "cache_hits_total": int(self._cache_hits_total),
                "cache_misses_total": int(self._cache_misses_total),
                "memories_total_by_agent": dict(self._memories_total_by_agent),
                "vectorless_total": int(self._vectorless_total),
                "embed_queue_depth": int(self._embed_queue_depth),
                "stage_buckets": {key: list(counts) for key, counts in self._stage_buckets.items()},
                "stage_sum": dict(self._stage_sum),
                "stage_count": dict(self._stage_count),
                "stage_attempts": dict(self._stage_attempts),
                "stage_retries": dict(self._stage_retries),
                "job_outcomes": dict(self._job_outcomes),
                "index_queue_depth": self._index_queue_depth,
                "index_inflight": self._index_inflight,
            }

    def render_prometheus(self) -> str:
        """Render snapshot in Prometheus exposition format (text/plain)."""
        snap = self.snapshot()
        lines: list[str] = []

        lines.append("# HELP agent_memory_requests_total Total HTTP requests processed.")
        lines.append("# TYPE agent_memory_requests_total counter")
        for (method, path, status), count in sorted(snap["requests_total"].items()):
            lines.append(
                "agent_memory_requests_total"
                f'{{method="{_label_escape(method)}",path="{_label_escape(path)}",status="{_label_escape(status)}"}} '
                f"{int(count)}"
            )

        lines.append("# HELP agent_memory_request_duration_seconds HTTP request latency in seconds.")
        lines.append("# TYPE agent_memory_request_duration_seconds histogram")
        duration_buckets: Dict[str, list[int]] = snap["request_duration_bucket_counts"]
        duration_sum: Dict[str, float] = snap["request_duration_sum"]
        duration_count: Dict[str, int] = snap["request_duration_count"]
        for path in sorted(duration_buckets.keys()):
            path_label = _label_escape(path)
            buckets = duration_buckets[path]
            for upper_bound, bucket_value in zip(self.REQUEST_DURATION_BUCKETS, buckets):
                lines.append(
                    "agent_memory_request_duration_seconds_bucket"
                    f'{{path="{path_label}",le="{_format_bucket(upper_bound)}"}} '
                    f"{int(bucket_value)}"
                )

            lines.append(
                "agent_memory_request_duration_seconds_bucket"
                f'{{path="{path_label}",le="+Inf"}} '
                f"{int(duration_count.get(path, 0))}"
            )
            lines.append(
                "agent_memory_request_duration_seconds_sum"
                f'{{path="{path_label}"}} '
                f"{_format_float(float(duration_sum.get(path, 0.0)))}"
            )
            lines.append(
                "agent_memory_request_duration_seconds_count"
                f'{{path="{path_label}"}} '
                f"{int(duration_count.get(path, 0))}"
            )

        lines.append("# HELP agent_memory_cache_hits_total Total search cache hits.")
        lines.append("# TYPE agent_memory_cache_hits_total counter")
        lines.append(f"agent_memory_cache_hits_total {snap['cache_hits_total']}")

        lines.append("# HELP agent_memory_cache_misses_total Total search cache misses.")
        lines.append("# TYPE agent_memory_cache_misses_total counter")
        lines.append(f"agent_memory_cache_misses_total {snap['cache_misses_total']}")

        lines.append("# HELP agent_memory_memories_total Total memories stored, by agent.")
        lines.append("# TYPE agent_memory_memories_total gauge")
        for agent, total in sorted(snap["memories_total_by_agent"].items()):
            lines.append(
                "agent_memory_memories_total"
                f'{{agent="{_label_escape(agent)}"}} '
                f"{int(total)}"
            )

        lines.append("# HELP agent_memory_vectorless_total Total memories without vectors.")
        lines.append("# TYPE agent_memory_vectorless_total gauge")
        lines.append(f"agent_memory_vectorless_total {int(snap['vectorless_total'])}")

        lines.append("# HELP agent_memory_embed_queue_depth Estimated embed worker queue depth.")
        lines.append("# TYPE agent_memory_embed_queue_depth gauge")
        lines.append(f"agent_memory_embed_queue_depth {int(snap['embed_queue_depth'])}")

        lines.append("# HELP agent_memory_stage_duration_seconds Bounded operation stage duration.")
        lines.append("# TYPE agent_memory_stage_duration_seconds histogram")
        for key, buckets in sorted(snap["stage_buckets"].items()):
            operation, stage, outcome = key
            labels = f'operation="{operation}",stage="{stage}",outcome="{outcome}"'
            for upper, count in zip(self.REQUEST_DURATION_BUCKETS, buckets):
                lines.append(f'agent_memory_stage_duration_seconds_bucket{{{labels},le="{_format_bucket(upper)}"}} {count}')
            lines.append(f'agent_memory_stage_duration_seconds_bucket{{{labels},le="+Inf"}} {snap["stage_count"][key]}')
            lines.append(f'agent_memory_stage_duration_seconds_sum{{{labels}}} {_format_float(snap["stage_sum"][key])}')
            lines.append(f'agent_memory_stage_duration_seconds_count{{{labels}}} {snap["stage_count"][key]}')
        for metric, values in (("stage_attempts", snap["stage_attempts"]),
                               ("stage_retries", snap["stage_retries"])):
            lines.append(f"# TYPE agent_memory_{metric}_total counter")
            for stage, count in sorted(values.items()):
                lines.append(f'agent_memory_{metric}_total{{stage="{stage}"}} {count}')
        lines.append("# TYPE agent_memory_index_jobs_total counter")
        for (stage, outcome), count in sorted(snap["job_outcomes"].items()):
            lines.append(f'agent_memory_index_jobs_total{{stage="{stage}",outcome="{outcome}"}} {count}')
        for name in ("index_queue_depth", "index_inflight"):
            lines.append(f"# TYPE agent_memory_{name} gauge")
            lines.append(f"agent_memory_{name} {snap[name]}")
        return "\n".join(lines) + "\n"


collector = MetricsCollector()


def _safe_duration(value: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid_metric_duration")
    return float(value)


_METRIC_PATHS = frozenset({
    "/", "/docs", "/openapi.json", "/redoc",
    *{"/v1/" + path for path in (
        "recall", "relearn-source", "capture/status", "capture", "store", "rule", "pin", "unpin",
        "forget", "search", "decay", "gc", "consolidate", "compress", "dashboard", "stats", "agents",
        "health", "health/deep", "health/live", "health/backend", "health/doctor", "metrics/lessons",
        "metrics/prometheus", "metrics", "export", "import", "amnesia-check", "solved", "admin/rotate-key",
    )},
})


def normalize_metric_path(path: str) -> str:
    """Never emit attacker-controlled paths or operation identities as labels."""
    if isinstance(path, str) and path.startswith("/v1/operations/"):
        return "/v1/operations/{request_id}"
    return path if isinstance(path, str) and path in _METRIC_PATHS else "/other"


def record_stage_metric(*, operation: str, stage: str, duration_seconds: float,
                        outcome: str = "completed") -> None:
    """Best-effort runtime observation; telemetry never changes product state."""
    try:
        collector.record_stage(operation, stage, duration_seconds, outcome)
    except Exception:
        return


def record_attempt_metric(stage: str, *, retry: bool = False) -> None:
    try:
        collector.record_attempt(stage, retry=retry)
    except Exception:
        return


def record_job_metric(stage: str, outcome: str) -> None:
    try:
        collector.record_job(stage, outcome)
    except Exception:
        return


def set_index_metric_gauges(*, queue_depth: int, inflight: int) -> None:
    try:
        collector.set_index_gauges(queue_depth=queue_depth, inflight=inflight)
    except Exception:
        return


def record_request_metric(*, method: str, path: str, status: int, duration_seconds: float) -> None:
    collector.record_request(method=method, path=path, status=status, duration_seconds=duration_seconds)


def record_cache_hit(count: int = 1) -> None:
    collector.inc_cache_hit(count=count)


def record_cache_miss(count: int = 1) -> None:
    collector.inc_cache_miss(count=count)


def set_runtime_gauges(
    *,
    memories_total_by_agent: Dict[str, int],
    vectorless_total: int,
    embed_queue_depth: int,
) -> None:
    collector.set_runtime_gauges(
        memories_total_by_agent=memories_total_by_agent,
        vectorless_total=vectorless_total,
        embed_queue_depth=embed_queue_depth,
    )


def render_prometheus_metrics() -> str:
    return collector.render_prometheus()


def reset_metrics() -> None:
    collector.reset()


def _label_escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_bucket(value: float) -> str:
    return f"{float(value):g}"


def _format_float(value: float) -> str:
    text = f"{float(value):.9f}".rstrip("0").rstrip(".")
    return text if text else "0"
