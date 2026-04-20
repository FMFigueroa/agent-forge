from anthropic import AsyncAnthropic

from agent_forge.agents.drafter import Drafter
from agent_forge.agents.orchestrator import Orchestrator
from agent_forge.agents.pipeline import GenerationPipeline
from agent_forge.agents.researcher import Researcher
from agent_forge.config import settings
from agent_forge.evals.judge import Judge
from agent_forge.evals.runner import EvalRunner
from agent_forge.observability.sqlite_tracer import SQLiteTracer


def build_tracer(db_url: str | None = None) -> SQLiteTracer:
    return SQLiteTracer(db_url or settings.db_url)


def build_pipeline(db_url: str | None = None) -> GenerationPipeline:
    tracer = build_tracer(db_url)
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return GenerationPipeline(
        researcher=Researcher(client=client, tracer=tracer),
        drafter=Drafter(client=client, tracer=tracer),
        orchestrator=Orchestrator(client=client, tracer=tracer),
        tracer=tracer,
    )


def build_eval_runner(db_url: str | None = None) -> tuple[EvalRunner, SQLiteTracer]:
    tracer = build_tracer(db_url)
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    pipeline = GenerationPipeline(
        researcher=Researcher(client=client, tracer=tracer),
        drafter=Drafter(client=client, tracer=tracer),
        orchestrator=Orchestrator(client=client, tracer=tracer),
        tracer=tracer,
    )
    judge = Judge(client=client, tracer=tracer)
    runner = EvalRunner(pipeline=pipeline, judge=judge, store=tracer)
    return runner, tracer
