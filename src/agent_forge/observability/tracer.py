from typing import Protocol

from agent_forge.agents.models import TraceSpan


class Tracer(Protocol):
    def record(self, span: TraceSpan) -> None: ...


class InMemoryTracer:
    def __init__(self) -> None:
        self.spans: list[TraceSpan] = []

    def record(self, span: TraceSpan) -> None:
        self.spans.append(span)

    def by_run(self, run_id: str) -> list[TraceSpan]:
        return [s for s in self.spans if s.run_id == run_id]

    def total_cost(self, run_id: str | None = None) -> float:
        spans = self.by_run(run_id) if run_id else self.spans
        return sum(s.cost_usd for s in spans)
