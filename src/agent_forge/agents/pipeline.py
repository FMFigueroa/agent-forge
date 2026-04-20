import uuid
from typing import Protocol

from pydantic import BaseModel

from agent_forge.agents.models import AgentResult
from agent_forge.observability.tracer import Tracer


class AgentRunner(Protocol):
    name: str

    async def run(self, user_message: str, *, run_id: str | None = None) -> AgentResult: ...


class GenerationResult(BaseModel):
    run_id: str
    topic: str
    research: str
    draft: str
    final_post: str
    total_cost: float


class GenerationPipeline:
    def __init__(
        self,
        *,
        researcher: AgentRunner,
        drafter: AgentRunner,
        orchestrator: AgentRunner,
        tracer: Tracer,
    ) -> None:
        self.researcher = researcher
        self.drafter = drafter
        self.orchestrator = orchestrator
        self.tracer = tracer

    async def run(self, topic: str) -> GenerationResult:
        run_id = str(uuid.uuid4())
        try:
            research = await self.researcher.run(topic, run_id=run_id)

            draft_input = f"Topic: {topic}\n\nResearch:\n{research.text}"
            draft = await self.drafter.run(draft_input, run_id=run_id)

            orch_input = f"Topic: {topic}\n\nResearch:\n{research.text}\n\nDraft:\n{draft.text}"
            final = await self.orchestrator.run(orch_input, run_id=run_id)
        except Exception:
            self.tracer.mark_run_status(run_id, "failed")
            raise

        self.tracer.mark_run_status(run_id, "completed")

        return GenerationResult(
            run_id=run_id,
            topic=topic,
            research=research.text,
            draft=draft.text,
            final_post=final.text,
            total_cost=research.span.cost_usd + draft.span.cost_usd + final.span.cost_usd,
        )
