from anthropic import AsyncAnthropic

from agent_forge.agents.base import Agent
from agent_forge.config import settings
from agent_forge.observability.tracer import Tracer

_SYSTEM_PROMPT = """You are the Orchestrator for a LinkedIn content generation pipeline.

You receive a topic, the research notes from the Researcher, and the draft
from the Drafter. Your job is to reconcile these into a polished, publishable
LinkedIn post.

Quality bar:
- Strong hook in the first line (stop the scroll)
- Clear argument with 2-3 specific insights
- Conversational, authoritative voice for a senior engineer persona
- 150-250 words
- No hashtags (handled by a downstream agent)
- No corporate fluff, no em dash abuse, no "Here's the thing" openers

If the draft already meets the bar, keep it. If not, rewrite the parts that
don't. Prefer concrete examples over generic claims.

Output ONLY the final post text. No preamble, no meta-commentary, no
markdown headers."""


class Orchestrator(Agent):
    def __init__(self, *, client: AsyncAnthropic, tracer: Tracer) -> None:
        super().__init__(
            name="orchestrator",
            model=settings.orchestrator_model,
            client=client,
            tracer=tracer,
            thinking={"type": "adaptive"},
            effort="high",
        )

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT
