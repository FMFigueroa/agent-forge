from typing import cast

from anthropic import AsyncAnthropic

from agent_forge.agents.drafter import Drafter
from agent_forge.agents.orchestrator import Orchestrator
from agent_forge.agents.researcher import Researcher
from agent_forge.observability.tracer import InMemoryTracer


def _dummy_client() -> AsyncAnthropic:
    return cast(AsyncAnthropic, object())


def test_researcher_wires_specialist_model_and_prompt() -> None:
    agent = Researcher(client=_dummy_client(), tracer=InMemoryTracer())

    assert agent.name == "researcher"
    assert agent.model == "claude-sonnet-4-6"
    assert agent.thinking is None
    assert agent.effort is None
    assert "Research Specialist" in agent.system_prompt
    assert "bullet" in agent.system_prompt.lower()


def test_drafter_wires_specialist_model_and_prompt() -> None:
    agent = Drafter(client=_dummy_client(), tracer=InMemoryTracer())

    assert agent.name == "drafter"
    assert agent.model == "claude-sonnet-4-6"
    assert agent.thinking is None
    assert agent.effort is None
    assert "LinkedIn" in agent.system_prompt
    assert "hook" in agent.system_prompt.lower()


def test_orchestrator_uses_opus_with_adaptive_thinking_and_high_effort() -> None:
    agent = Orchestrator(client=_dummy_client(), tracer=InMemoryTracer())

    assert agent.name == "orchestrator"
    assert agent.model == "claude-opus-4-7"
    assert agent.thinking == {"type": "adaptive"}
    assert agent.effort == "high"
    assert "Orchestrator" in agent.system_prompt
    assert "reconcile" in agent.system_prompt.lower()
