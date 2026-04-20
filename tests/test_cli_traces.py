from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_forge.agents.models import TokenUsage, TraceSpan
from agent_forge.cli import app
from agent_forge.config import settings
from agent_forge.observability.factory import build_tracer
from agent_forge.observability.sqlite_tracer import SQLiteTracer

runner = CliRunner(env={"COLUMNS": "200"})


def _make_span(
    *,
    span_id: str,
    run_id: str,
    agent_name: str = "drafter",
    cost_usd: float = 0.5,
    latency_ms: float = 120.0,
    stop_reason: str | None = "end_turn",
    error: str | None = None,
    started_offset_sec: float = 0.0,
) -> TraceSpan:
    started = datetime.now(UTC) + timedelta(seconds=started_offset_sec)
    return TraceSpan(
        span_id=span_id,
        run_id=run_id,
        agent_name=agent_name,
        model="claude-haiku-4-5",
        started_at=started,
        ended_at=started + timedelta(milliseconds=latency_ms),
        latency_ms=latency_ms,
        usage=TokenUsage(input_tokens=100, output_tokens=50, cache_read_input_tokens=30),
        cost_usd=cost_usd,
        stop_reason=stop_reason,
        error=error,
    )


@pytest.fixture
def db_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    monkeypatch.setattr(settings, "db_url", url)
    return url


def test_build_tracer_uses_settings_db_url_by_default(db_url: str) -> None:
    tracer = build_tracer()
    assert isinstance(tracer, SQLiteTracer)


def test_build_tracer_override_wins_over_settings(tmp_path: Path) -> None:
    override = f"sqlite:///{tmp_path / 'override.db'}"
    tracer = build_tracer(db_url=override)
    assert isinstance(tracer, SQLiteTracer)


def test_traces_ls_shows_empty_message_when_no_runs(db_url: str) -> None:
    result = runner.invoke(app, ["traces", "ls"])
    assert result.exit_code == 0
    assert "No runs recorded yet" in result.stdout


def test_traces_ls_renders_recent_runs(db_url: str) -> None:
    tracer = SQLiteTracer(db_url)
    tracer.record(_make_span(span_id="s1", run_id="run-older", cost_usd=0.10, started_offset_sec=0))
    tracer.record(_make_span(span_id="s2", run_id="run-newer", cost_usd=0.25, started_offset_sec=5))
    tracer.record(_make_span(span_id="s3", run_id="run-newer", cost_usd=0.15, started_offset_sec=6))

    result = runner.invoke(app, ["traces", "ls"])

    assert result.exit_code == 0
    assert "run-newer" in result.stdout
    assert "run-older" in result.stdout
    newer_idx = result.stdout.index("run-newer")
    older_idx = result.stdout.index("run-older")
    assert newer_idx < older_idx
    assert "$0.4000" in result.stdout
    assert "$0.1000" in result.stdout


def test_traces_ls_respects_limit(db_url: str) -> None:
    tracer = SQLiteTracer(db_url)
    for i in range(5):
        tracer.record(_make_span(span_id=f"s{i}", run_id=f"run-{i}", started_offset_sec=i))

    result = runner.invoke(app, ["traces", "ls", "--limit", "2"])

    assert result.exit_code == 0
    assert "run-4" in result.stdout
    assert "run-3" in result.stdout
    assert "run-0" not in result.stdout


def test_traces_show_returns_error_for_unknown_run(db_url: str) -> None:
    result = runner.invoke(app, ["traces", "show", "does-not-exist"])

    assert result.exit_code == 1
    assert "No spans found" in result.stdout


def test_traces_show_renders_spans_and_totals(db_url: str) -> None:
    tracer = SQLiteTracer(db_url)
    tracer.record(_make_span(span_id="s1", run_id="run-A", agent_name="researcher"))
    tracer.record(_make_span(span_id="s2", run_id="run-A", agent_name="drafter"))

    result = runner.invoke(app, ["traces", "show", "run-A"])

    assert result.exit_code == 0
    assert "run-A" in result.stdout
    assert "researcher" in result.stdout
    assert "drafter" in result.stdout
    assert "spans: 2" in result.stdout
    assert "$1.0000" in result.stdout


def test_traces_show_marks_error_spans(db_url: str) -> None:
    tracer = SQLiteTracer(db_url)
    tracer.record(
        _make_span(
            span_id="boom",
            run_id="run-E",
            stop_reason=None,
            error="RuntimeError: API down",
            cost_usd=0.0,
        )
    )

    result = runner.invoke(app, ["traces", "show", "run-E"])

    assert result.exit_code == 0
    assert "RuntimeError" in result.stdout
