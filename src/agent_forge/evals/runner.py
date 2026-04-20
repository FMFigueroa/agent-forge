from typing import Protocol

from pydantic import BaseModel

from agent_forge.agents.pipeline import GenerationResult
from agent_forge.evals.models import Judgment


class PipelineRunner(Protocol):
    async def run(self, topic: str) -> GenerationResult: ...


class JudgeRunner(Protocol):
    async def evaluate(self, *, topic: str, post: str, run_id: str) -> Judgment: ...


class JudgmentStore(Protocol):
    def record_judgment(self, judgment: Judgment) -> None: ...


class EvalItem(BaseModel):
    topic: str
    generation: GenerationResult
    judgment: Judgment


class EvalReport(BaseModel):
    items: list[EvalItem]

    @property
    def mean_overall(self) -> float:
        if not self.items:
            return 0.0
        return sum(i.judgment.scores.overall for i in self.items) / len(self.items)

    @property
    def mean_hook(self) -> float:
        if not self.items:
            return 0.0
        return sum(i.judgment.scores.hook_strength for i in self.items) / len(self.items)

    @property
    def worst_item(self) -> EvalItem | None:
        if not self.items:
            return None
        return min(self.items, key=lambda i: i.judgment.scores.overall)


class EvalRunner:
    def __init__(
        self,
        *,
        pipeline: PipelineRunner,
        judge: JudgeRunner,
        store: JudgmentStore,
    ) -> None:
        self.pipeline = pipeline
        self.judge = judge
        self.store = store

    async def run_topic(self, topic: str) -> EvalItem:
        generation = await self.pipeline.run(topic)
        judgment = await self.judge.evaluate(
            topic=topic,
            post=generation.final_post,
            run_id=generation.run_id,
        )
        self.store.record_judgment(judgment)
        return EvalItem(topic=topic, generation=generation, judgment=judgment)

    async def run_set(self, topics: list[str]) -> EvalReport:
        items = [await self.run_topic(t) for t in topics]
        return EvalReport(items=items)
