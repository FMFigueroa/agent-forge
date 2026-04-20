from datetime import UTC, datetime

import pytest

from agent_forge.agents.image_prompter import CarouselSlide
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


class FakeHashtagPicker:
    def __init__(
        self,
        tags: list[str],
        tracer: InMemoryTracer,
        cost: float = 0.05,
        name: str = "hashtag_specialist",
    ) -> None:
        self.name = name
        self.tags = tags
        self.tracer = tracer
        self.cost = cost
        self.calls: list[tuple[str, str]] = []
        self._should_raise: Exception | None = None

    def will_raise(self, exc: Exception) -> None:
        self._should_raise = exc

    async def pick(self, *, post: str, run_id: str) -> list[str]:
        self.calls.append((post, run_id))
        if self._should_raise is not None:
            raise self._should_raise
        self.tracer.record(
            TraceSpan(
                span_id=f"span-{self.name}-{len(self.calls)}",
                run_id=run_id,
                agent_name=self.name,
                model="fake",
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                latency_ms=1.0,
                cost_usd=self.cost,
            )
        )
        return self.tags


class FakeCarouselDesigner:
    def __init__(
        self,
        slides: list[CarouselSlide],
        tracer: InMemoryTracer,
        cost: float = 0.05,
        name: str = "image_prompter",
    ) -> None:
        self.name = name
        self.slides = slides
        self.tracer = tracer
        self.cost = cost
        self.calls: list[tuple[str, str]] = []

    async def design(self, *, post: str, run_id: str) -> list[CarouselSlide]:
        self.calls.append((post, run_id))
        self.tracer.record(
            TraceSpan(
                span_id=f"span-{self.name}-{len(self.calls)}",
                run_id=run_id,
                agent_name=self.name,
                model="fake",
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                latency_ms=1.0,
                cost_usd=self.cost,
            )
        )
        return self.slides


def _make_pipeline(
    *,
    researcher_cost: float = 0.10,
    drafter_cost: float = 0.20,
    editor_cost: float = 0.15,
    orchestrator_cost: float = 0.50,
    hashtag_cost: float = 0.02,
    image_cost: float = 0.03,
) -> tuple[
    GenerationPipeline,
    FakeAgent,
    FakeAgent,
    FakeAgent,
    FakeAgent,
    FakeHashtagPicker,
    FakeCarouselDesigner,
    InMemoryTracer,
]:
    tracer = InMemoryTracer()
    researcher = FakeAgent("researcher", "- insight 1\n- insight 2", tracer, researcher_cost)
    drafter = FakeAgent("drafter", "Draft post text.", tracer, drafter_cost)
    editor = FakeAgent("editor", "Edited draft text.", tracer, editor_cost)
    orchestrator = FakeAgent("orchestrator", "Final polished post.", tracer, orchestrator_cost)
    hashtag_specialist = FakeHashtagPicker(["#AI", "#LLMs", "#Engineering"], tracer, hashtag_cost)
    image_prompter = FakeCarouselDesigner(
        [
            CarouselSlide(title="Hook", image_prompt="prompt 1"),
            CarouselSlide(title="Insight", image_prompt="prompt 2"),
            CarouselSlide(title="Takeaway", image_prompt="prompt 3"),
        ],
        tracer,
        image_cost,
    )
    pipeline = GenerationPipeline(
        researcher=researcher,
        drafter=drafter,
        editor=editor,
        orchestrator=orchestrator,
        hashtag_specialist=hashtag_specialist,
        image_prompter=image_prompter,
        tracer=tracer,
    )
    return (
        pipeline,
        researcher,
        drafter,
        editor,
        orchestrator,
        hashtag_specialist,
        image_prompter,
        tracer,
    )


async def test_pipeline_runs_all_six_agents_with_shared_run_id() -> None:
    (
        pipeline,
        researcher,
        drafter,
        editor,
        orchestrator,
        hashtag,
        images,
        _,
    ) = _make_pipeline()

    result = await pipeline.run("context windows")

    assert result.topic == "context windows"
    assert result.research == "- insight 1\n- insight 2"
    assert result.draft == "Draft post text."
    assert result.edited_draft == "Edited draft text."
    assert result.final_post == "Final polished post."
    assert result.hashtags == ["#AI", "#LLMs", "#Engineering"]
    assert len(result.carousel_slides) == 3

    assert [c[1] for c in researcher.calls] == [result.run_id]
    assert [c[1] for c in drafter.calls] == [result.run_id]
    assert [c[1] for c in editor.calls] == [result.run_id]
    assert [c[1] for c in orchestrator.calls] == [result.run_id]
    assert [c[1] for c in hashtag.calls] == [result.run_id]
    assert [c[1] for c in images.calls] == [result.run_id]


async def test_pipeline_threads_outputs_between_agents() -> None:
    pipeline, _, drafter, editor, orchestrator, hashtag, images, _ = _make_pipeline()

    await pipeline.run("context windows")

    drafter_input = drafter.calls[0][0]
    assert "context windows" in drafter_input
    assert "- insight 1" in drafter_input

    editor_input = editor.calls[0][0]
    assert "Draft post text." in editor_input

    orch_input = orchestrator.calls[0][0]
    assert "Edited draft text." in orch_input
    assert "- insight 1" in orch_input

    # Hashtag + image prompter see the final polished post, not the draft
    assert hashtag.calls[0][0] == "Final polished post."
    assert images.calls[0][0] == "Final polished post."


async def test_pipeline_total_cost_includes_all_six_spans() -> None:
    pipeline, *_ = _make_pipeline(
        researcher_cost=0.10,
        drafter_cost=0.20,
        editor_cost=0.15,
        orchestrator_cost=0.50,
        hashtag_cost=0.02,
        image_cost=0.03,
    )

    result = await pipeline.run("topic")

    assert result.total_cost == pytest.approx(1.00)


async def test_pipeline_marks_run_completed_on_success() -> None:
    pipeline, *_, tracer = _make_pipeline()

    result = await pipeline.run("topic")

    assert tracer.run_statuses[result.run_id] == "completed"


async def test_pipeline_marks_run_failed_and_reraises_on_error() -> None:
    pipeline, _, drafter, *_, tracer = _make_pipeline()
    drafter.will_raise(RuntimeError("API timeout"))

    with pytest.raises(RuntimeError, match="API timeout"):
        await pipeline.run("topic")

    assert len(tracer.run_statuses) == 1
    [(_, status)] = tracer.run_statuses.items()
    assert status == "failed"


async def test_pipeline_marks_failed_if_post_processor_raises() -> None:
    pipeline, *_, hashtag, _, tracer = _make_pipeline()
    hashtag.will_raise(ValueError("bad json"))

    with pytest.raises(ValueError, match="bad json"):
        await pipeline.run("topic")

    [(_, status)] = tracer.run_statuses.items()
    assert status == "failed"


async def test_pipeline_generates_unique_run_ids_per_call() -> None:
    pipeline, *_ = _make_pipeline()

    r1 = await pipeline.run("topic A")
    r2 = await pipeline.run("topic B")

    assert r1.run_id != r2.run_id
