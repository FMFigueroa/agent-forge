from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agent_forge.agents.models import TokenUsage, TraceSpan
from agent_forge.evals.models import Judgment, RubricScore
from agent_forge.observability.sqlite_tracer import SQLiteTracer


def _seed_run(tracer: SQLiteTracer, run_id: str) -> None:
    tracer.record(
        TraceSpan(
            span_id=f"span-{run_id}",
            run_id=run_id,
            agent_name="orchestrator",
            model="claude-opus-4-7",
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            latency_ms=10.0,
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            cost_usd=0.0,
        )
    )


def _make_judgment(
    run_id: str,
    *,
    hook: int = 8,
    clarity: int = 7,
    persona: int = 9,
    engage: int = 6,
    reasoning: str = "reasoning",
    offset_sec: float = 0.0,
) -> Judgment:
    return Judgment(
        run_id=run_id,
        judge_model="claude-sonnet-4-6",
        scores=RubricScore(
            hook_strength=hook,
            clarity=clarity,
            persona_fit=persona,
            engagement_potential=engage,
        ),
        reasoning=reasoning,
        judged_at=datetime.now(UTC) + timedelta(seconds=offset_sec),
    )


@pytest.fixture
def tracer(tmp_path: Path) -> SQLiteTracer:
    return SQLiteTracer(f"sqlite:///{tmp_path / 'j.db'}")


def test_record_and_retrieve_judgment(tracer: SQLiteTracer) -> None:
    _seed_run(tracer, "run-A")
    tracer.record_judgment(_make_judgment("run-A"))

    [judgment] = tracer.judgments_for_run("run-A")

    assert judgment.run_id == "run-A"
    assert judgment.scores.hook_strength == 8
    assert judgment.scores.clarity == 7
    assert judgment.scores.overall == pytest.approx(7.5)
    assert judgment.judge_model == "claude-sonnet-4-6"
    assert judgment.reasoning == "reasoning"


def test_judgments_for_run_ordered_by_judged_at(tracer: SQLiteTracer) -> None:
    _seed_run(tracer, "run-A")
    tracer.record_judgment(_make_judgment("run-A", reasoning="late", offset_sec=10))
    tracer.record_judgment(_make_judgment("run-A", reasoning="early", offset_sec=0))

    judgments = tracer.judgments_for_run("run-A")

    assert [j.reasoning for j in judgments] == ["early", "late"]


def test_judgments_for_run_isolates_runs(tracer: SQLiteTracer) -> None:
    _seed_run(tracer, "run-A")
    _seed_run(tracer, "run-B")
    tracer.record_judgment(_make_judgment("run-A"))
    tracer.record_judgment(_make_judgment("run-B"))

    assert len(tracer.judgments_for_run("run-A")) == 1
    assert len(tracer.judgments_for_run("run-B")) == 1
    assert tracer.judgments_for_run("nonexistent") == []


def test_overall_is_persisted_as_computed(tracer: SQLiteTracer) -> None:
    _seed_run(tracer, "run-A")
    tracer.record_judgment(_make_judgment("run-A", hook=10, clarity=10, persona=10, engage=10))

    [j] = tracer.judgments_for_run("run-A")
    assert j.scores.overall == 10.0
