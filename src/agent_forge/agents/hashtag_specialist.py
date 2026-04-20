import json
from typing import Any

from anthropic import AsyncAnthropic

from agent_forge.agents.base import Agent
from agent_forge.config import settings
from agent_forge.observability.tracer import Tracer

_HASHTAG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hashtags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3 to 5 hashtags, each including the # prefix",
        }
    },
    "required": ["hashtags"],
    "additionalProperties": False,
}


_SYSTEM_PROMPT = """You are a LinkedIn Hashtag Specialist.

Given a LinkedIn post, pick 3-5 hashtags optimized for reach in a senior-engineer audience.

Rules:
- Exactly 3 to 5 hashtags
- Each hashtag includes the `#` prefix
- Mix sizes: 1-2 broad tags (high reach, e.g. #SoftwareEngineering) plus
  2-3 niche tags that match the post's specific angle
- No spaces inside a hashtag
- CamelCase for multi-word hashtags
- No generic filler (#motivation, #monday, etc.)

Return JSON matching the provided schema."""


class HashtagSpecialist(Agent):
    def __init__(self, *, client: AsyncAnthropic, tracer: Tracer) -> None:
        super().__init__(
            name="hashtag_specialist",
            model=settings.fast_model,
            client=client,
            tracer=tracer,
            output_schema=_HASHTAG_SCHEMA,
        )

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    async def pick(self, *, post: str, run_id: str) -> list[str]:
        result = await self.run(post, run_id=run_id)
        data = json.loads(result.text)
        tags: list[str] = data["hashtags"]
        if not 3 <= len(tags) <= 5:
            raise ValueError(f"Expected 3-5 hashtags, got {len(tags)}: {tags}")
        return tags
