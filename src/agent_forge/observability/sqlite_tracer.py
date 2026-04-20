from statistics import quantiles

from sqlalchemy import Engine
from sqlmodel import Session, col, select

from agent_forge.agents.models import TraceSpan
from agent_forge.observability.db import RunRow, SpanRow, create_db_engine, init_db


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

    def mark_run_status(self, run_id: str, status: str) -> None:
        with Session(self._engine) as session:
            run = session.get(RunRow, run_id)
            if run is None:
                return
            run.status = status
            session.add(run)
            session.commit()
