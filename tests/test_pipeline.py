from datetime import UTC, datetime

import pytest

from agent_forge.agents.models import AgentResult, TraceSpan
from agent_forge.agents.pipeline import GenerationPipeline
from agent_forge.observability.tracer import InMemoryTracer


class FakeAgent:
    def __init__(self, name: str, response: str, tracer: InMemoryTracer, cost: float = 0.5) -> None:
        self.name = name
        self.response = response
        self.tracer = tracer
        self.cost = cost
        self.calls: list[tuple[str, str | None]] = []
        self._should_raise: Exception | None = None

    def will_raise(self, exc: Exception) -> None:
        self._should_raise = exc

    async def run(self, user_message: str, *, run_id: str | None = None) -> AgentResult:
        self.calls.append((user_message, run_id))
        if self._should_raise is not None:
            raise self._should_raise
        span = TraceSpan(
            span_id=f"span-{self.name}-{len(self.calls)}",
            run_id=run_id or "default",
            agent_name=self.name,
            model="fake",
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            latency_ms=1.0,
            cost_usd=self.cost,
        )
        self.tracer.record(span)
        return AgentResult(text=self.response, span=span)


def _make_pipeline(
    *,
    researcher_text: str = "- insight 1\n- insight 2",
    drafter_text: str = "Draft post text.",
    orchestrator_text: str = "Final polished post.",
    researcher_cost: float = 0.10,
    drafter_cost: float = 0.20,
    orchestrator_cost: float = 0.50,
) -> tuple[GenerationPipeline, FakeAgent, FakeAgent, FakeAgent, InMemoryTracer]:
    tracer = InMemoryTracer()
    researcher = FakeAgent("researcher", researcher_text, tracer, researcher_cost)
    drafter = FakeAgent("drafter", drafter_text, tracer, drafter_cost)
    orchestrator = FakeAgent("orchestrator", orchestrator_text, tracer, orchestrator_cost)
    pipeline = GenerationPipeline(
        researcher=researcher,
        drafter=drafter,
        orchestrator=orchestrator,
        tracer=tracer,
    )
    return pipeline, researcher, drafter, orchestrator, tracer


async def test_pipeline_runs_agents_in_order_with_shared_run_id() -> None:
    pipeline, researcher, drafter, orchestrator, _ = _make_pipeline()

    result = await pipeline.run("context windows")

    assert result.topic == "context windows"
    assert result.research == "- insight 1\n- insight 2"
    assert result.draft == "Draft post text."
    assert result.final_post == "Final polished post."

    assert len(researcher.calls) == 1
    assert researcher.calls[0][0] == "context windows"

    assert len(drafter.calls) == 1
    drafter_input, _ = drafter.calls[0]
    assert "context windows" in drafter_input
    assert "- insight 1" in drafter_input

    assert len(orchestrator.calls) == 1
    orch_input, _ = orchestrator.calls[0]
    assert "Draft post text." in orch_input
    assert "- insight 1" in orch_input

    run_ids = {
        researcher.calls[0][1],
        drafter.calls[0][1],
        orchestrator.calls[0][1],
    }
    assert run_ids == {result.run_id}


async def test_pipeline_aggregates_total_cost() -> None:
    pipeline, *_ = _make_pipeline(researcher_cost=0.10, drafter_cost=0.20, orchestrator_cost=0.50)

    result = await pipeline.run("topic")

    assert result.total_cost == pytest.approx(0.80)


async def test_pipeline_marks_run_completed_on_success() -> None:
    pipeline, *_, tracer = _make_pipeline()

    result = await pipeline.run("topic")

    assert tracer.run_statuses[result.run_id] == "completed"


async def test_pipeline_marks_run_failed_and_reraises_on_error() -> None:
    pipeline, _, drafter, _, tracer = _make_pipeline()
    drafter.will_raise(RuntimeError("API timeout"))

    with pytest.raises(RuntimeError, match="API timeout"):
        await pipeline.run("topic")

    # Drafter raised, so exactly one run_id got registered (by researcher)
    assert len(tracer.run_statuses) == 1
    [(_, status)] = tracer.run_statuses.items()
    assert status == "failed"


async def test_pipeline_generates_unique_run_ids_per_call() -> None:
    pipeline, *_ = _make_pipeline()

    r1 = await pipeline.run("topic A")
    r2 = await pipeline.run("topic B")

    assert r1.run_id != r2.run_id
