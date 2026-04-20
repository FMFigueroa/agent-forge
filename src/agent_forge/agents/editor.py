from anthropic import AsyncAnthropic

from agent_forge.agents.base import Agent
from agent_forge.config import settings
from agent_forge.observability.tracer import Tracer

_SYSTEM_PROMPT = """You are a LinkedIn Editor for a senior engineer persona.

You receive a topic and the Drafter's first pass. Refine it:
- Sharpen the hook — the first line must earn the next line
- Tighten structure — every paragraph must pay rent
- Fix the voice — conversational, authoritative, no hedging, no corporate tone
- Cut anything generic — replace abstractions with concrete details from the draft
- Keep 150-250 words
- No hashtags (downstream agent handles those)

If the draft already meets the bar, keep it. Rewrite only what doesn't land.

Output ONLY the refined post. No meta-commentary, no diff, no explanation."""


class Editor(Agent):
    def __init__(self, *, client: AsyncAnthropic, tracer: Tracer) -> None:
        super().__init__(
            name="editor",
            model=settings.specialist_model,
            client=client,
            tracer=tracer,
        )

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT
