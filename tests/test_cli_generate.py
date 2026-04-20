from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from agent_forge.agents.models import AgentResult, TraceSpan
from agent_forge.agents.pipeline import GenerationResult
from agent_forge.cli import app

runner = CliRunner(env={"COLUMNS": "200"})


def _fake_generation_result(run_id: str = "run-abc123") -> GenerationResult:
    from agent_forge.agents.image_prompter import CarouselSlide

    return GenerationResult(
        run_id=run_id,
        topic="context windows",
        research="- bullet 1\n- bullet 2",
        draft="Draft body.",
        edited_draft="Refined draft body.",
        final_post="Final post body — polished and ready to ship.",
        hashtags=["#AI", "#LLMs", "#Engineering"],
        carousel_slides=[
            CarouselSlide(title="Hook", image_prompt="close-up of a circuit board"),
            CarouselSlide(title="Insight", image_prompt="split-screen diagram"),
            CarouselSlide(title="Takeaway", image_prompt="minimalist text card"),
        ],
        total_cost=0.1234,
    )


def test_generate_runs_pipeline_and_prints_final_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pipeline = AsyncMock()
    fake_pipeline.run = AsyncMock(return_value=_fake_generation_result())

    monkeypatch.setattr("agent_forge.cli.build_pipeline", lambda *a, **kw: fake_pipeline)

    result = runner.invoke(app, ["generate", "context windows"])

    assert result.exit_code == 0
    assert "Final post body" in result.stdout
    assert "run-abc123" in result.stdout
    assert "$0.1234" in result.stdout
    assert "traces show run-abc123" in result.stdout
    fake_pipeline.run.assert_awaited_once_with("context windows")


def test_generate_propagates_pipeline_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pipeline = AsyncMock()
    fake_pipeline.run = AsyncMock(side_effect=RuntimeError("API down"))

    monkeypatch.setattr("agent_forge.cli.build_pipeline", lambda *a, **kw: fake_pipeline)

    result = runner.invoke(app, ["generate", "topic"])

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)


def test_agent_result_and_trace_span_shape() -> None:
    """Smoke: GenerationResult composes AgentResult/TraceSpan cleanly."""
    span = TraceSpan(
        span_id="s1",
        run_id="r1",
        agent_name="researcher",
        model="claude-sonnet-4-6",
        started_at=datetime.now(UTC),
        ended_at=datetime.now(UTC),
        latency_ms=10.0,
        cost_usd=0.05,
    )
    agent_result = AgentResult(text="hi", span=span)
    assert agent_result.text == "hi"
    assert agent_result.span.agent_name == "researcher"
