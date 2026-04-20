from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from agent_forge.agents.pipeline import GenerationResult
from agent_forge.cli import app
from agent_forge.config import settings
from agent_forge.evals.models import Judgment, RubricScore
from agent_forge.evals.runner import EvalItem, EvalReport
from agent_forge.observability.sqlite_tracer import SQLiteTracer

runner = CliRunner(env={"COLUMNS": "200"})


def _item(run_id: str, topic: str, overall: int) -> EvalItem:
    return EvalItem(
        topic=topic,
        generation=GenerationResult(
            run_id=run_id,
            topic=topic,
            research="- insight",
            draft="draft",
            edited_draft="edited",
            final_post="final",
            hashtags=["#AI"],
            carousel_slides=[],
            total_cost=0.1,
        ),
        judgment=Judgment(
            run_id=run_id,
            judge_model="claude-sonnet-4-6",
            scores=RubricScore(
                hook_strength=overall,
                clarity=overall,
                persona_fit=overall,
                engagement_potential=overall,
            ),
            reasoning=f"reasoning for {run_id}",
            judged_at=datetime.now(UTC),
        ),
    )


@pytest.fixture
def db_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    url = f"sqlite:///{tmp_path / 'evals.db'}"
    monkeypatch.setattr(settings, "db_url", url)
    return url


def test_evals_run_prints_per_topic_and_aggregate_scores(
    monkeypatch: pytest.MonkeyPatch, db_url: str
) -> None:
    fake_tracer = SQLiteTracer(db_url)
    fake_runner = AsyncMock()
    fake_runner.run_set = AsyncMock(
        return_value=EvalReport(
            items=[
                _item("run-1", "topic one", overall=8),
                _item("run-2", "topic two", overall=4),
            ]
        )
    )
    monkeypatch.setattr(
        "agent_forge.cli.build_eval_runner",
        lambda *a, **kw: (fake_runner, fake_tracer),
    )

    result = runner.invoke(app, ["evals", "run", "--limit", "2"])

    assert result.exit_code == 0
    assert "topic one" in result.stdout
    assert "topic two" in result.stdout
    assert "run-1" in result.stdout
    assert "run-2" in result.stdout
    assert "Mean overall" in result.stdout
    assert "6.00" in result.stdout
    assert "Worst" in result.stdout
    assert "topic two" in result.stdout


def test_evals_run_with_default_uses_full_golden_set(
    monkeypatch: pytest.MonkeyPatch, db_url: str
) -> None:
    from agent_forge.evals.golden_set import GOLDEN_TOPICS

    fake_tracer = SQLiteTracer(db_url)
    fake_runner = AsyncMock()
    fake_runner.run_set = AsyncMock(return_value=EvalReport(items=[]))
    monkeypatch.setattr(
        "agent_forge.cli.build_eval_runner",
        lambda *a, **kw: (fake_runner, fake_tracer),
    )

    result = runner.invoke(app, ["evals", "run"])

    assert result.exit_code == 0
    fake_runner.run_set.assert_awaited_once_with(GOLDEN_TOPICS)


def test_evals_show_errors_when_no_judgments(db_url: str) -> None:
    result = runner.invoke(app, ["evals", "show", "nonexistent"])

    assert result.exit_code == 1
    assert "No judgments found" in result.stdout


def test_evals_show_renders_judgment_panel(db_url: str) -> None:
    tracer = SQLiteTracer(db_url)
    # Seed a span so the run exists, then record a judgment
    from agent_forge.agents.models import TokenUsage, TraceSpan

    tracer.record(
        TraceSpan(
            span_id="s1",
            run_id="run-XYZ",
            agent_name="orchestrator",
            model="claude-opus-4-7",
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            latency_ms=10.0,
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            cost_usd=0.0,
        )
    )
    tracer.record_judgment(
        Judgment(
            run_id="run-XYZ",
            judge_model="claude-sonnet-4-6",
            scores=RubricScore(hook_strength=9, clarity=8, persona_fit=7, engagement_potential=6),
            reasoning="Hook lands hard, engagement falls off at the end.",
            judged_at=datetime.now(UTC),
        )
    )

    result = runner.invoke(app, ["evals", "show", "run-XYZ"])

    assert result.exit_code == 0
    assert "run-XYZ" in result.stdout
    assert "claude-sonnet-4-6" in result.stdout
    assert "Hook lands hard" in result.stdout
    assert "overall" in result.stdout
    assert "7.5" in result.stdout
