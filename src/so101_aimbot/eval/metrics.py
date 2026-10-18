"""Session metrics for Gridshot evaluation (manual or OCR later)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SessionMetrics:
    """Accumulated stats comparable to AimCrane's post-round summary."""

    score: int = 0
    hits: int = 0
    misses: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def mean_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    def record_hit(self, latency_ms: float) -> None:
        self.hits += 1
        self.latencies_ms.append(latency_ms)

    def record_miss(self) -> None:
        self.misses += 1
