from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agent_forge.agents.models import TokenUsage, TraceSpan
from agent_forge.observability.sqlite_tracer import SQLiteTracer


def _make_span(
    *,
    span_id: str,
    run_id: str = "run-1",
    agent_name: str = "drafter",
    model: str = "claude-haiku-4-5",
    latency_ms: float = 100.0,
    cost_usd: float = 0.5,
    input_tokens: int = 10,
    output_tokens: int = 5,
    started_offset_sec: float = 0.0,
    stop_reason: str | None = "end_turn",
    error: str | None = None,
) -> TraceSpan:
    started = datetime.now(UTC) + timedelta(seconds=started_offset_sec)
    return TraceSpan(
        span_id=span_id,
        run_id=run_id,
        agent_name=agent_name,
        model=model,
        started_at=started,
        ended_at=started + timedelta(milliseconds=latency_ms),
        latency_ms=latency_ms,
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        cost_usd=cost_usd,
        stop_reason=stop_reason,
        error=error,
    )


@pytest.fixture
def tracer(tmp_path: Path) -> SQLiteTracer:
    return SQLiteTracer(f"sqlite:///{tmp_path / 'trace.db'}")


def test_record_persists_span_and_creates_run(tracer: SQLiteTracer) -> None:
    span = _make_span(span_id="s1", run_id="run-A")
    tracer.record(span)

    retrieved = tracer.by_run("run-A")
    assert len(retrieved) == 1
    assert retrieved[0].span_id == "s1"
    assert retrieved[0].agent_name == "drafter"
    assert retrieved[0].usage.input_tokens == 10


def test_by_run_returns_spans_ordered_by_start_time(tracer: SQLiteTracer) -> None:
    tracer.record(_make_span(span_id="late", started_offset_sec=5))
    tracer.record(_make_span(span_id="early", started_offset_sec=0))
    tracer.record(_make_span(span_id="middle", started_offset_sec=2))

    ids = [s.span_id for s in tracer.by_run("run-1")]
    assert ids == ["early", "middle", "late"]


def test_by_run_isolates_runs(tracer: SQLiteTracer) -> None:
    tracer.record(_make_span(span_id="s1", run_id="run-A"))
    tracer.record(_make_span(span_id="s2", run_id="run-B"))

    assert len(tracer.by_run("run-A")) == 1
    assert len(tracer.by_run("run-B")) == 1
    assert tracer.by_run("nonexistent") == []


def test_total_cost_aggregates_across_and_filters_by_run(tracer: SQLiteTracer) -> None:
    tracer.record(_make_span(span_id="s1", run_id="run-A", cost_usd=0.10))
    tracer.record(_make_span(span_id="s2", run_id="run-A", cost_usd=0.25))
    tracer.record(_make_span(span_id="s3", run_id="run-B", cost_usd=1.00))

    assert tracer.total_cost() == pytest.approx(1.35)
    assert tracer.total_cost(run_id="run-A") == pytest.approx(0.35)
    assert tracer.total_cost(run_id="run-B") == pytest.approx(1.00)


def test_latency_p95_returns_none_when_no_data(tracer: SQLiteTracer) -> None:
    assert tracer.latency_p95("drafter") is None


def test_latency_p95_with_single_sample(tracer: SQLiteTracer) -> None:
    tracer.record(_make_span(span_id="s1", agent_name="drafter", latency_ms=42.0))
    assert tracer.latency_p95("drafter") == 42.0


def test_latency_p95_scopes_to_agent_name(tracer: SQLiteTracer) -> None:
    for i in range(20):
        tracer.record(_make_span(span_id=f"d{i}", agent_name="drafter", latency_ms=float(i + 1)))
    tracer.record(_make_span(span_id="other", agent_name="editor", latency_ms=99999.0))

    p95 = tracer.latency_p95("drafter")
    assert p95 is not None
    assert 18.0 <= p95 <= 20.0


def test_record_handles_error_spans(tracer: SQLiteTracer) -> None:
    tracer.record(
        _make_span(
            span_id="boom",
            cost_usd=0.0,
            stop_reason=None,
            error="RuntimeError: API down",
        )
    )

    [span] = tracer.by_run("run-1")
    assert span.error == "RuntimeError: API down"
    assert span.stop_reason is None


def test_mark_run_status_updates_existing_run(tracer: SQLiteTracer) -> None:
    from sqlmodel import Session

    from agent_forge.observability.db import RunRow

    tracer.record(_make_span(span_id="s1", run_id="run-X"))
    tracer.mark_run_status("run-X", "completed")

    with Session(tracer._engine) as session:
        run = session.get(RunRow, "run-X")
        assert run is not None
        assert run.status == "completed"


def test_mark_run_status_noop_for_unknown_run(tracer: SQLiteTracer) -> None:
    tracer.mark_run_status("does-not-exist", "completed")


def test_persists_across_tracer_instances(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    t1 = SQLiteTracer(db_url)
    t1.record(_make_span(span_id="s1", run_id="run-P"))

    t2 = SQLiteTracer(db_url)
    assert len(t2.by_run("run-P")) == 1
