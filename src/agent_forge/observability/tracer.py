from typing import Protocol

from agent_forge.agents.models import TraceSpan


class Tracer(Protocol):
    def record(self, span: TraceSpan) -> None: ...
    def mark_run_status(self, run_id: str, status: str) -> None: ...
    def by_run(self, run_id: str) -> list[TraceSpan]: ...


class InMemoryTracer:
    def __init__(self) -> None:
        self.spans: list[TraceSpan] = []
        self.run_statuses: dict[str, str] = {}

    def record(self, span: TraceSpan) -> None:
        self.spans.append(span)
        self.run_statuses.setdefault(span.run_id, "running")

    def mark_run_status(self, run_id: str, status: str) -> None:
        if run_id in self.run_statuses:
            self.run_statuses[run_id] = status

    def by_run(self, run_id: str) -> list[TraceSpan]:
        return [s for s in self.spans if s.run_id == run_id]

    def total_cost(self, run_id: str | None = None) -> float:
        spans = self.by_run(run_id) if run_id else self.spans
        return sum(s.cost_usd for s in spans)
