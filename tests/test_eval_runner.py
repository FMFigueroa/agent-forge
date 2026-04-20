from datetime import UTC, datetime

import pytest

from agent_forge.agents.models import AgentResult, TraceSpan
from agent_forge.agents.pipeline import GenerationResult
from agent_forge.evals.models import Judgment, RubricScore
from agent_forge.evals.runner import EvalRunner


class _FakePipeline:
    def __init__(self, posts: list[str]) -> None:
        self._posts = posts
        self.calls: list[str] = []

    async def run(self, topic: str) -> GenerationResult:
        self.calls.append(topic)
        idx = len(self.calls) - 1
        return GenerationResult(
            run_id=f"run-{idx}",
            topic=topic,
            research="- insight",
            draft="draft",
            edited_draft="edited",
            final_post=self._posts[idx],
            hashtags=["#AI"],
            carousel_slides=[],
            total_cost=0.1,
        )


class _FakeJudge:
    def __init__(self, scores: list[RubricScore]) -> None:
        self._scores = scores
        self.calls: list[tuple[str, str, str]] = []

    async def evaluate(self, *, topic: str, post: str, run_id: str) -> Judgment:
        self.calls.append((topic, post, run_id))
        idx = len(self.calls) - 1
        return Judgment(
            run_id=run_id,
            judge_model="claude-sonnet-4-6",
            scores=self._scores[idx],
            reasoning=f"reasoning {idx}",
            judged_at=datetime.now(UTC),
        )


class _FakeStore:
    def __init__(self) -> None:
        self.recorded: list[Judgment] = []

    def record_judgment(self, judgment: Judgment) -> None:
        self.recorded.append(judgment)


def _score(overall: int) -> RubricScore:
    return RubricScore(
        hook_strength=overall,
        clarity=overall,
        persona_fit=overall,
        engagement_potential=overall,
    )


async def test_run_topic_chains_pipeline_judge_and_store() -> None:
    pipeline = _FakePipeline(posts=["Final post."])
    judge = _FakeJudge(scores=[_score(7)])
    store = _FakeStore()
    runner = EvalRunner(pipeline=pipeline, judge=judge, store=store)

    item = await runner.run_topic("context windows")

    assert item.topic == "context windows"
    assert item.generation.run_id == "run-0"
    assert item.judgment.run_id == "run-0"
    assert item.judgment.scores.overall == 7.0
    assert pipeline.calls == ["context windows"]
    assert judge.calls == [("context windows", "Final post.", "run-0")]
    assert len(store.recorded) == 1


async def test_run_set_produces_report_with_aggregates() -> None:
    pipeline = _FakePipeline(posts=["p0", "p1", "p2"])
    judge = _FakeJudge(scores=[_score(10), _score(4), _score(7)])
    store = _FakeStore()
    runner = EvalRunner(pipeline=pipeline, judge=judge, store=store)

    report = await runner.run_set(["t0", "t1", "t2"])

    assert len(report.items) == 3
    assert report.mean_overall == pytest.approx((10 + 4 + 7) / 3)
    assert report.mean_hook == pytest.approx((10 + 4 + 7) / 3)
    assert len(store.recorded) == 3

    worst = report.worst_item
    assert worst is not None
    assert worst.topic == "t1"
    assert worst.judgment.scores.overall == 4.0


async def test_run_set_empty_returns_empty_report() -> None:
    runner = EvalRunner(
        pipeline=_FakePipeline(posts=[]),
        judge=_FakeJudge(scores=[]),
        store=_FakeStore(),
    )

    report = await runner.run_set([])

    assert report.items == []
    assert report.mean_overall == 0.0
    assert report.worst_item is None


def test_agent_result_shape_with_judgment() -> None:
    """Smoke: Judgment composes cleanly alongside generation AgentResult."""
    span = TraceSpan(
        span_id="s1",
        run_id="r1",
        agent_name="judge",
        model="claude-sonnet-4-6",
        started_at=datetime.now(UTC),
        ended_at=datetime.now(UTC),
        latency_ms=10.0,
        cost_usd=0.01,
    )
    AgentResult(text='{"ok": true}', span=span)
