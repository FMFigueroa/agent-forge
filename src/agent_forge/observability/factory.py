from anthropic import AsyncAnthropic

from agent_forge.agents.drafter import Drafter
from agent_forge.agents.editor import Editor
from agent_forge.agents.hashtag_specialist import HashtagSpecialist
from agent_forge.agents.image_prompter import ImagePrompter
from agent_forge.agents.orchestrator import Orchestrator
from agent_forge.agents.pipeline import GenerationPipeline
from agent_forge.agents.researcher import Researcher
from agent_forge.config import settings
from agent_forge.evals.judge import Judge
from agent_forge.evals.runner import EvalRunner
from agent_forge.observability.sqlite_tracer import SQLiteTracer


def build_tracer(db_url: str | None = None) -> SQLiteTracer:
    return SQLiteTracer(db_url or settings.db_url)


def _build_pipeline_with(client: AsyncAnthropic, tracer: SQLiteTracer) -> GenerationPipeline:
    return GenerationPipeline(
        researcher=Researcher(client=client, tracer=tracer),
        drafter=Drafter(client=client, tracer=tracer),
        editor=Editor(client=client, tracer=tracer),
        orchestrator=Orchestrator(client=client, tracer=tracer),
        hashtag_specialist=HashtagSpecialist(client=client, tracer=tracer),
        image_prompter=ImagePrompter(client=client, tracer=tracer),
        tracer=tracer,
    )


def build_pipeline(db_url: str | None = None) -> GenerationPipeline:
    tracer = build_tracer(db_url)
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _build_pipeline_with(client, tracer)


def build_eval_runner(db_url: str | None = None) -> tuple[EvalRunner, SQLiteTracer]:
    tracer = build_tracer(db_url)
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    pipeline = _build_pipeline_with(client, tracer)
    judge = Judge(client=client, tracer=tracer)
    runner = EvalRunner(pipeline=pipeline, judge=judge, store=tracer)
    return runner, tracer
