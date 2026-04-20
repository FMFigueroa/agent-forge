from typing import Any, cast

import pytest
from anthropic import AsyncAnthropic

from agent_forge.agents.base import Agent, compute_cost
from agent_forge.agents.models import TokenUsage
from agent_forge.observability.tracer import InMemoryTracer


class _FakeTextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeUsage:
    def __init__(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class _FakeMessage:
    def __init__(self, text: str, usage: _FakeUsage, stop_reason: str = "end_turn") -> None:
        self.content = [_FakeTextBlock(text)]
        self.usage = usage
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, response: _FakeMessage | Exception) -> None:
        self._response = response
        self.call_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _FakeMessage:
        self.call_kwargs = kwargs
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _FakeClient:
    def __init__(self, response: _FakeMessage | Exception) -> None:
        self.messages = _FakeMessages(response)


class EchoAgent(Agent):
    @property
    def system_prompt(self) -> str:
        return "You echo the user input."


def _make_agent(
    client: _FakeClient, model: str = "claude-haiku-4-5"
) -> tuple[EchoAgent, InMemoryTracer]:
    tracer = InMemoryTracer()
    agent = EchoAgent(
        name="echo",
        model=model,
        client=cast(AsyncAnthropic, client),
        tracer=tracer,
    )
    return agent, tracer


async def test_run_returns_text_and_emits_span() -> None:
    client = _FakeClient(_FakeMessage("hola bro", _FakeUsage(input_tokens=100, output_tokens=50)))
    agent, tracer = _make_agent(client)

    result = await agent.run("qué hay")

    assert result.text == "hola bro"
    assert len(tracer.spans) == 1
    span = tracer.spans[0]
    assert span.agent_name == "echo"
    assert span.model == "claude-haiku-4-5"
    assert span.stop_reason == "end_turn"
    assert span.usage.input_tokens == 100
    assert span.usage.output_tokens == 50
    assert span.error is None
    assert span.latency_ms >= 0


async def test_run_computes_cost_for_haiku() -> None:
    client = _FakeClient(
        _FakeMessage("ok", _FakeUsage(input_tokens=1_000_000, output_tokens=1_000_000))
    )
    agent, tracer = _make_agent(client, model="claude-haiku-4-5")

    await agent.run("hi")

    assert tracer.spans[0].cost_usd == pytest.approx(1.0 + 5.0)


async def test_run_enables_prompt_caching_on_system() -> None:
    client = _FakeClient(_FakeMessage("ok", _FakeUsage(input_tokens=10, output_tokens=5)))
    agent, _ = _make_agent(client)

    await agent.run("hi")

    assert client.messages.call_kwargs is not None
    system = client.messages.call_kwargs["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert system[0]["text"] == "You echo the user input."


async def test_run_propagates_run_id_across_spans() -> None:
    client = _FakeClient(_FakeMessage("ok", _FakeUsage(input_tokens=1, output_tokens=1)))
    agent, tracer = _make_agent(client)

    await agent.run("a", run_id="run-123")
    await agent.run("b", run_id="run-123")

    assert [s.run_id for s in tracer.spans] == ["run-123", "run-123"]
    assert len(tracer.by_run("run-123")) == 2


async def test_run_records_span_on_error_and_reraises() -> None:
    boom = RuntimeError("API down")
    client = _FakeClient(boom)
    agent, tracer = _make_agent(client)

    with pytest.raises(RuntimeError, match="API down"):
        await agent.run("hi")

    assert len(tracer.spans) == 1
    span = tracer.spans[0]
    assert span.error is not None
    assert "RuntimeError" in span.error
    assert "API down" in span.error
    assert span.usage.total_tokens == 0
    assert span.cost_usd == 0.0


def test_compute_cost_includes_cache_write_and_read() -> None:
    usage = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    cost = compute_cost("claude-opus-4-7", usage)

    assert cost == pytest.approx(5.0 + 25.0 + 5.0 * 1.25 + 5.0 * 0.1)


def test_compute_cost_returns_zero_for_unknown_model() -> None:
    usage = TokenUsage(input_tokens=1000, output_tokens=1000)
    assert compute_cost("not-a-real-model", usage) == 0.0


def test_in_memory_tracer_total_cost() -> None:
    tracer = InMemoryTracer()
    from datetime import UTC, datetime

    from agent_forge.agents.models import TraceSpan

    for i in range(3):
        tracer.record(
            TraceSpan(
                span_id=f"s{i}",
                run_id="r1" if i < 2 else "r2",
                agent_name="x",
                model="claude-haiku-4-5",
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                latency_ms=0.0,
                cost_usd=1.5,
            )
        )

    assert tracer.total_cost() == pytest.approx(4.5)
    assert tracer.total_cost(run_id="r1") == pytest.approx(3.0)
    assert tracer.total_cost(run_id="r2") == pytest.approx(1.5)
