import asyncio
import uuid
from typing import Protocol

from pydantic import BaseModel

from agent_forge.agents.image_prompter import CarouselSlide
from agent_forge.agents.models import AgentResult
from agent_forge.observability.tracer import Tracer


class AgentRunner(Protocol):
    name: str

    async def run(self, user_message: str, *, run_id: str | None = None) -> AgentResult: ...


class HashtagPicker(Protocol):
    name: str

    async def pick(self, *, post: str, run_id: str) -> list[str]: ...


class CarouselDesigner(Protocol):
    name: str

    async def design(self, *, post: str, run_id: str) -> list[CarouselSlide]: ...


class GenerationResult(BaseModel):
    run_id: str
    topic: str
    research: str
    draft: str
    edited_draft: str
    final_post: str
    hashtags: list[str]
    carousel_slides: list[CarouselSlide]
    total_cost: float


class GenerationPipeline:
    def __init__(
        self,
        *,
        researcher: AgentRunner,
        drafter: AgentRunner,
        editor: AgentRunner,
        orchestrator: AgentRunner,
        hashtag_specialist: HashtagPicker,
        image_prompter: CarouselDesigner,
        tracer: Tracer,
    ) -> None:
        self.researcher = researcher
        self.drafter = drafter
        self.editor = editor
        self.orchestrator = orchestrator
        self.hashtag_specialist = hashtag_specialist
        self.image_prompter = image_prompter
        self.tracer = tracer

    async def run(self, topic: str) -> GenerationResult:
        run_id = str(uuid.uuid4())
        try:
            research = await self.researcher.run(topic, run_id=run_id)

            draft_input = f"Topic: {topic}\n\nResearch:\n{research.text}"
            draft = await self.drafter.run(draft_input, run_id=run_id)

            edit_input = f"Topic: {topic}\n\nDraft:\n{draft.text}"
            edited = await self.editor.run(edit_input, run_id=run_id)

            orch_input = (
                f"Topic: {topic}\n\nResearch:\n{research.text}\n\nEdited draft:\n{edited.text}"
            )
            final = await self.orchestrator.run(orch_input, run_id=run_id)

            hashtags, slides = await asyncio.gather(
                self.hashtag_specialist.pick(post=final.text, run_id=run_id),
                self.image_prompter.design(post=final.text, run_id=run_id),
            )
        except Exception:
            self.tracer.mark_run_status(run_id, "failed")
            raise

        self.tracer.mark_run_status(run_id, "completed")

        total_cost = sum(s.cost_usd for s in self.tracer.by_run(run_id))
        return GenerationResult(
            run_id=run_id,
            topic=topic,
            research=research.text,
            draft=draft.text,
            edited_draft=edited.text,
            final_post=final.text,
            hashtags=hashtags,
            carousel_slides=slides,
            total_cost=total_cost,
        )
