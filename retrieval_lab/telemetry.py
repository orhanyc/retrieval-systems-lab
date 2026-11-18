from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from statistics import mean
from threading import RLock
from time import monotonic


@dataclass(frozen=True)
class SearchTrace:
    trace_id: str
    method: str
    query_length: int
    results: int
    latency_ms: float
    cache_hit: bool
    top_score: float


class RetrievalTelemetry:
    def __init__(self, capacity: int = 1000) -> None:
        self.traces: deque[SearchTrace] = deque(maxlen=capacity)
        self.lock = RLock()

    def record(self, trace: SearchTrace) -> None:
        with self.lock:
            self.traces.append(trace)

    def snapshot(self) -> dict:
        with self.lock:
            traces = list(self.traces)
        latencies = sorted(trace.latency_ms for trace in traces)
        percentile_index = min(len(latencies) - 1, int(len(latencies) * 0.95)) if latencies else 0
        return {
            "requests": len(traces),
            "methods": dict(Counter(trace.method for trace in traces)),
            "cache_hits": sum(trace.cache_hit for trace in traces),
            "empty_results": sum(trace.results == 0 for trace in traces),
            "mean_latency_ms": mean(latencies) if latencies else 0,
            "p95_latency_ms": latencies[percentile_index] if latencies else 0,
            "mean_top_score": mean(trace.top_score for trace in traces) if traces else 0,
        }


class SearchCache:
    def __init__(self, ttl_seconds: int = 300, capacity: int = 500) -> None:
        self.ttl_seconds = ttl_seconds
        self.capacity = capacity
        self.entries: dict[str, tuple[float, object]] = {}
        self.order: deque[str] = deque()
        self.lock = RLock()

    def get(self, key: str):
        with self.lock:
            item = self.entries.get(key)
            if not item:
                return None
            created, value = item
            if monotonic() - created > self.ttl_seconds:
                self.entries.pop(key, None)
                return None
            return value

    def put(self, key: str, value: object) -> None:
        with self.lock:
            if key not in self.entries:
                self.order.append(key)
            self.entries[key] = (monotonic(), value)
            while len(self.entries) > self.capacity:
                oldest = self.order.popleft()
                self.entries.pop(oldest, None)

    def invalidate(self) -> None:
        with self.lock:
            self.entries.clear()
            self.order.clear()
