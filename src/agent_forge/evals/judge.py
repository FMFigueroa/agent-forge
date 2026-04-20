import json
from datetime import UTC, datetime
from typing import Any

from anthropic import AsyncAnthropic

from agent_forge.agents.base import Agent
from agent_forge.config import settings
from agent_forge.evals.models import Judgment, RubricScore
from agent_forge.observability.tracer import Tracer

_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hook_strength": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "description": "Does the first line stop the scroll? 0 = generic, 10 = impossible to ignore",
        },
        "clarity": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "description": "Is the argument easy to follow? 0 = muddled, 10 = crystal clear",
        },
        "persona_fit": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "description": "Does it sound like a senior engineer? 0 = corporate fluff, 10 = authentic voice",
        },
        "engagement_potential": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "description": "Would LinkedIn readers save/share/comment? 0 = scroll past, 10 = must-engage",
        },
        "reasoning": {
            "type": "string",
            "description": "2-4 sentences explaining the scores, citing specific lines from the post",
        },
    },
    "required": [
        "hook_strength",
        "clarity",
        "persona_fit",
        "engagement_potential",
        "reasoning",
    ],
    "additionalProperties": False,
}


_SYSTEM_PROMPT = """You are a strict Evaluator of LinkedIn content for a senior engineer persona.

Given a topic and the post that was generated for it, score the post on four axes (0-10 each):

- hook_strength: Does the first line stop the scroll?
- clarity: Is the argument easy to follow?
- persona_fit: Does it sound like a senior engineer, not marketing copy?
- engagement_potential: Would LinkedIn readers actually save/share/comment?

Be honest and specific. A 10 means publishable as-is; a 5 means needs significant rework;
a 2 means fundamentally broken. Do not inflate scores to be polite.

Cite specific lines in your reasoning. Return JSON matching the provided schema."""


class Judge(Agent):
    def __init__(self, *, client: AsyncAnthropic, tracer: Tracer) -> None:
        super().__init__(
            name="judge",
            model=settings.specialist_model,
            client=client,
            tracer=tracer,
            output_schema=_JUDGE_SCHEMA,
        )

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    async def evaluate(self, *, topic: str, post: str, run_id: str) -> Judgment:
        user_message = f"Topic: {topic}\n\nPost:\n{post}"
        result = await self.run(user_message, run_id=run_id)
        data = json.loads(result.text)
        return Judgment(
            run_id=run_id,
            judge_model=self.model,
            scores=RubricScore(
                hook_strength=data["hook_strength"],
                clarity=data["clarity"],
                persona_fit=data["persona_fit"],
                engagement_potential=data["engagement_potential"],
            ),
            reasoning=data["reasoning"],
            judged_at=datetime.now(UTC),
        )
