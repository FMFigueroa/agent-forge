import json
from typing import Any

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from agent_forge.agents.base import Agent
from agent_forge.config import settings
from agent_forge.observability.tracer import Tracer


class CarouselSlide(BaseModel):
    title: str
    image_prompt: str


_CAROUSEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "slides": {
            "type": "array",
            "description": "3 to 4 carousel slides in order, title + image_prompt each",
            "items": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short headline for the slide, under 8 words",
                    },
                    "image_prompt": {
                        "type": "string",
                        "description": "Concrete visual prompt for an image generator, "
                        "including composition, subject, style",
                    },
                },
                "required": ["title", "image_prompt"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["slides"],
    "additionalProperties": False,
}


_SYSTEM_PROMPT = """You are a LinkedIn Image Prompter for carousel posts.

Given the final LinkedIn post, produce 3-4 carousel slides that visually
reinforce the post's argument:
- Slide 1 is always the hook — visual echo of the opening line
- Middle slides break down the key insights
- Last slide is the takeaway / call-to-think

Each slide has:
- title: a short headline under 8 words
- image_prompt: concrete visual prompt for an image generator. Include
  composition (close-up, wide, split-screen), subject (what's shown),
  style (photorealistic, flat illustration, diagram). No people with faces.

Return JSON matching the provided schema."""


class ImagePrompter(Agent):
    def __init__(self, *, client: AsyncAnthropic, tracer: Tracer) -> None:
        super().__init__(
            name="image_prompter",
            model=settings.fast_model,
            client=client,
            tracer=tracer,
            output_schema=_CAROUSEL_SCHEMA,
        )

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    async def design(self, *, post: str, run_id: str) -> list[CarouselSlide]:
        result = await self.run(post, run_id=run_id)
        data = json.loads(result.text)
        slides = [CarouselSlide(**s) for s in data["slides"]]
        if not 3 <= len(slides) <= 4:
            raise ValueError(f"Expected 3-4 slides, got {len(slides)}")
        return slides
