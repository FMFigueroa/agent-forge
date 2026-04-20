from anthropic import AsyncAnthropic

from agent_forge.agents.base import Agent
from agent_forge.config import settings
from agent_forge.observability.tracer import Tracer

_SYSTEM_PROMPT = """You are a Research Specialist for a LinkedIn content pipeline.

Given a topic, produce 5-7 concise bullets capturing:
- Key facts, frameworks, or concepts relevant to the topic
- Common misconceptions or surprising angles worth challenging
- Specific technical details or examples that make a post credible

Keep each bullet under 20 words. No preamble, no meta-commentary.

Output format (literal):
- bullet 1
- bullet 2
- bullet 3
..."""


class Researcher(Agent):
    def __init__(self, *, client: AsyncAnthropic, tracer: Tracer) -> None:
        super().__init__(
            name="researcher",
            model=settings.specialist_model,
            client=client,
            tracer=tracer,
        )

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT
