import uuid
from datetime import datetime
from statistics import quantiles

from pydantic import BaseModel
from sqlalchemy import Engine
from sqlmodel import Session, col, select

from agent_forge.agents.models import TraceSpan
from agent_forge.evals.models import Judgment, RubricScore
from agent_forge.observability.db import (
    JudgmentRow,
    RunRow,
    SpanRow,
    create_db_engine,
    init_db,
)


class RunSummary(BaseModel):
    run_id: str
    created_at: datetime
    status: str
    span_count: int
    total_cost: float


class SQLiteTracer:
    def __init__(self, db_url: str) -> None:
        self._engine: Engine = create_db_engine(db_url)
        init_db(self._engine)

    def record(self, span: TraceSpan) -> None:
        with Session(self._engine) as session:
            if session.get(RunRow, span.run_id) is None:
                session.add(RunRow(id=span.run_id, created_at=span.started_at))
            session.add(SpanRow.from_span(span))
            session.commit()

    def by_run(self, run_id: str) -> list[TraceSpan]:
        with Session(self._engine) as session:
            stmt = select(SpanRow).where(SpanRow.run_id == run_id).order_by(col(SpanRow.started_at))
            rows = session.scalars(stmt).all()
            return [r.to_span() for r in rows]

    def total_cost(self, run_id: str | None = None) -> float:
        with Session(self._engine) as session:
            stmt = select(SpanRow)
            if run_id is not None:
                stmt = stmt.where(SpanRow.run_id == run_id)
            rows = session.scalars(stmt).all()
            return sum(r.cost_usd for r in rows)

    def latency_p95(self, agent_name: str) -> float | None:
        with Session(self._engine) as session:
            stmt = select(SpanRow).where(SpanRow.agent_name == agent_name)
            latencies = [r.latency_ms for r in session.scalars(stmt).all()]
        if not latencies:
            return None
        if len(latencies) < 2:
            return latencies[0]
        return quantiles(latencies, n=20)[18]

    def list_runs(self, limit: int = 10) -> list[RunSummary]:
        with Session(self._engine) as session:
            runs = session.scalars(
                select(RunRow).order_by(col(RunRow.created_at).desc()).limit(limit)
            ).all()
            if not runs:
                return []
            run_ids = [r.id for r in runs]
            spans = session.scalars(select(SpanRow).where(col(SpanRow.run_id).in_(run_ids))).all()

        counts: dict[str, int] = {}
        costs: dict[str, float] = {}
        for s in spans:
            counts[s.run_id] = counts.get(s.run_id, 0) + 1
            costs[s.run_id] = costs.get(s.run_id, 0.0) + s.cost_usd

        return [
            RunSummary(
                run_id=r.id,
                created_at=r.created_at,
                status=r.status,
                span_count=counts.get(r.id, 0),
                total_cost=costs.get(r.id, 0.0),
            )
            for r in runs
        ]

    def mark_run_status(self, run_id: str, status: str) -> None:
        with Session(self._engine) as session:
            run = session.get(RunRow, run_id)
            if run is None:
                return
            run.status = status
            session.add(run)
            session.commit()

    def record_judgment(self, judgment: Judgment) -> None:
        with Session(self._engine) as session:
            session.add(
                JudgmentRow(
                    id=str(uuid.uuid4()),
                    run_id=judgment.run_id,
                    judge_model=judgment.judge_model,
                    hook_strength=judgment.scores.hook_strength,
                    clarity=judgment.scores.clarity,
                    persona_fit=judgment.scores.persona_fit,
                    engagement_potential=judgment.scores.engagement_potential,
                    overall=judgment.scores.overall,
                    reasoning=judgment.reasoning,
                    judged_at=judgment.judged_at,
                )
            )
            session.commit()

    def judgments_for_run(self, run_id: str) -> list[Judgment]:
        with Session(self._engine) as session:
            stmt = (
                select(JudgmentRow)
                .where(JudgmentRow.run_id == run_id)
                .order_by(col(JudgmentRow.judged_at))
            )
            rows = session.scalars(stmt).all()
        return [
            Judgment(
                run_id=r.run_id,
                judge_model=r.judge_model,
                scores=RubricScore(
                    hook_strength=r.hook_strength,
                    clarity=r.clarity,
                    persona_fit=r.persona_fit,
                    engagement_potential=r.engagement_potential,
                ),
                reasoning=r.reasoning,
                judged_at=r.judged_at,
            )
            for r in rows
        ]
