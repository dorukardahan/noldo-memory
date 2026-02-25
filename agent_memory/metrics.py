"""In-process metrics collector with Prometheus text exposition."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Dict, Tuple


class MetricsCollector:
    """Thread-safe metrics collector for API/recall instrumentation."""

    REQUEST_DURATION_BUCKETS = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
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

    def record_request(self, method: str, path: str, status: int, duration_seconds: float) -> None:
        """Record request counter and latency histogram observation."""
        method_norm = (method or "GET").upper()
        path_norm = path or "/"
        status_norm = str(status)
        duration = max(0.0, float(duration_seconds))

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
            }

    def render_prometheus(self) -> str:
        """Render snapshot in Prometheus exposition format (text/plain)."""
        snap = self.snapshot()
        lines: list[str] = []

        lines.append("# HELP asuman_memory_requests_total Total HTTP requests processed.")
        lines.append("# TYPE asuman_memory_requests_total counter")
        for (method, path, status), count in sorted(snap["requests_total"].items()):
            lines.append(
                "asuman_memory_requests_total"
                f'{{method="{_label_escape(method)}",path="{_label_escape(path)}",status="{_label_escape(status)}"}} '
                f"{int(count)}"
            )

        lines.append("# HELP asuman_memory_request_duration_seconds HTTP request latency in seconds.")
        lines.append("# TYPE asuman_memory_request_duration_seconds histogram")
        duration_buckets: Dict[str, list[int]] = snap["request_duration_bucket_counts"]
        duration_sum: Dict[str, float] = snap["request_duration_sum"]
        duration_count: Dict[str, int] = snap["request_duration_count"]
        for path in sorted(duration_buckets.keys()):
            path_label = _label_escape(path)
            buckets = duration_buckets[path]
            for upper_bound, bucket_value in zip(self.REQUEST_DURATION_BUCKETS, buckets):
                lines.append(
                    "asuman_memory_request_duration_seconds_bucket"
                    f'{{path="{path_label}",le="{_format_bucket(upper_bound)}"}} '
                    f"{int(bucket_value)}"
                )

            lines.append(
                "asuman_memory_request_duration_seconds_bucket"
                f'{{path="{path_label}",le="+Inf"}} '
                f"{int(duration_count.get(path, 0))}"
            )
            lines.append(
                "asuman_memory_request_duration_seconds_sum"
                f'{{path="{path_label}"}} '
                f"{_format_float(float(duration_sum.get(path, 0.0)))}"
            )
            lines.append(
                "asuman_memory_request_duration_seconds_count"
                f'{{path="{path_label}"}} '
                f"{int(duration_count.get(path, 0))}"
            )

        lines.append("# HELP asuman_memory_cache_hits_total Total search cache hits.")
        lines.append("# TYPE asuman_memory_cache_hits_total counter")
        lines.append(f"asuman_memory_cache_hits_total {snap['cache_hits_total']}")

        lines.append("# HELP asuman_memory_cache_misses_total Total search cache misses.")
        lines.append("# TYPE asuman_memory_cache_misses_total counter")
        lines.append(f"asuman_memory_cache_misses_total {snap['cache_misses_total']}")

        lines.append("# HELP asuman_memory_memories_total Total memories stored, by agent.")
        lines.append("# TYPE asuman_memory_memories_total gauge")
        for agent, total in sorted(snap["memories_total_by_agent"].items()):
            lines.append(
                "asuman_memory_memories_total"
                f'{{agent="{_label_escape(agent)}"}} '
                f"{int(total)}"
            )

        lines.append("# HELP asuman_memory_vectorless_total Total memories without vectors.")
        lines.append("# TYPE asuman_memory_vectorless_total gauge")
        lines.append(f"asuman_memory_vectorless_total {int(snap['vectorless_total'])}")

        lines.append("# HELP asuman_memory_embed_queue_depth Estimated embed worker queue depth.")
        lines.append("# TYPE asuman_memory_embed_queue_depth gauge")
        lines.append(f"asuman_memory_embed_queue_depth {int(snap['embed_queue_depth'])}")

        return "\n".join(lines) + "\n"


collector = MetricsCollector()


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
