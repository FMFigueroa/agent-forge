import json
from typing import Any, cast

import pytest
from anthropic import AsyncAnthropic

from agent_forge.evals.judge import Judge
from agent_forge.evals.models import RubricScore
from agent_forge.observability.tracer import InMemoryTracer


class _FakeTextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeUsage:
    def __init__(self) -> None:
        self.input_tokens = 10
        self.output_tokens = 5
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeTextBlock(text)]
        self.usage = _FakeUsage()
        self.stop_reason = "end_turn"


class _FakeMessages:
    def __init__(self, text: str) -> None:
        self._text = text
        self.call_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _FakeMessage:
        self.call_kwargs = kwargs
        return _FakeMessage(self._text)


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.messages = _FakeMessages(text)


def _valid_judge_json() -> str:
    return json.dumps(
        {
            "hook_strength": 8,
            "clarity": 7,
            "persona_fit": 9,
            "engagement_potential": 6,
            "reasoning": "Hook is specific and technical. Clarity suffers in the third paragraph.",
        }
    )


def test_rubric_score_overall_averages_four_dimensions() -> None:
    s = RubricScore(hook_strength=8, clarity=6, persona_fit=10, engagement_potential=4)
    assert s.overall == pytest.approx(7.0)


def test_rubric_score_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        RubricScore(hook_strength=11, clarity=5, persona_fit=5, engagement_potential=5)


async def test_judge_passes_structured_output_schema_to_api() -> None:
    client = _FakeClient(_valid_judge_json())
    tracer = InMemoryTracer()
    judge = Judge(client=cast(AsyncAnthropic, client), tracer=tracer)

    await judge.evaluate(topic="topic", post="post body", run_id="run-1")

    kwargs = client.messages.call_kwargs
    assert kwargs is not None
    assert "output_config" in kwargs
    fmt = kwargs["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert "hook_strength" in fmt["schema"]["properties"]


async def test_judge_parses_json_response_into_judgment() -> None:
    client = _FakeClient(_valid_judge_json())
    tracer = InMemoryTracer()
    judge = Judge(client=cast(AsyncAnthropic, client), tracer=tracer)

    judgment = await judge.evaluate(topic="t", post="p", run_id="run-abc")

    assert judgment.run_id == "run-abc"
    assert judgment.judge_model == "claude-sonnet-4-6"
    assert judgment.scores.hook_strength == 8
    assert judgment.scores.clarity == 7
    assert judgment.scores.persona_fit == 9
    assert judgment.scores.engagement_potential == 6
    assert judgment.scores.overall == pytest.approx(7.5)
    assert "Hook is specific" in judgment.reasoning


async def test_judge_emits_span_under_shared_run_id() -> None:
    client = _FakeClient(_valid_judge_json())
    tracer = InMemoryTracer()
    judge = Judge(client=cast(AsyncAnthropic, client), tracer=tracer)

    await judge.evaluate(topic="t", post="p", run_id="shared-run")

    assert len(tracer.spans) == 1
    assert tracer.spans[0].run_id == "shared-run"
    assert tracer.spans[0].agent_name == "judge"
