from anthropic import AsyncAnthropic

from agent_forge.agents.base import Agent
from agent_forge.config import settings
from agent_forge.observability.tracer import Tracer

_SYSTEM_PROMPT = """You are a LinkedIn Copywriter writing for a senior engineer persona.

Given a topic and research bullets, write a draft post that:
- Opens with a hook that stops the scroll (specific claim, contrarian take, or concrete scenario)
- Builds a clear argument with 2-3 specific insights from the research
- Uses a conversational, authoritative voice — no corporate fluff
- Lands between 150-250 words
- Avoids hashtags (another agent handles those)
- Avoids em dash abuse and "Here's the thing" openers

Output ONLY the draft text. No headers, no preamble, no meta-commentary."""


class Drafter(Agent):
    def __init__(self, *, client: AsyncAnthropic, tracer: Tracer) -> None:
        super().__init__(
            name="drafter",
            model=settings.specialist_model,
            client=client,
            tracer=tracer,
        )

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT
