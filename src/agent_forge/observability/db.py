from datetime import UTC, datetime

from sqlalchemy import Engine
from sqlmodel import Field, SQLModel, create_engine

from agent_forge.agents.models import TokenUsage, TraceSpan


class RunRow(SQLModel, table=True):
    __tablename__ = "runs"  # type: ignore[assignment]

    id: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: str = "running"


class SpanRow(SQLModel, table=True):
    __tablename__ = "spans"  # type: ignore[assignment]

    span_id: str = Field(primary_key=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    agent_name: str = Field(index=True)
    model: str
    started_at: datetime
    ended_at: datetime
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    cost_usd: float
    stop_reason: str | None = None
    error: str | None = None

    @classmethod
    def from_span(cls, span: TraceSpan) -> "SpanRow":
        return cls(
            span_id=span.span_id,
            run_id=span.run_id,
            agent_name=span.agent_name,
            model=span.model,
            started_at=span.started_at,
            ended_at=span.ended_at,
            latency_ms=span.latency_ms,
            input_tokens=span.usage.input_tokens,
            output_tokens=span.usage.output_tokens,
            cache_creation_input_tokens=span.usage.cache_creation_input_tokens,
            cache_read_input_tokens=span.usage.cache_read_input_tokens,
            cost_usd=span.cost_usd,
            stop_reason=span.stop_reason,
            error=span.error,
        )

    def to_span(self) -> TraceSpan:
        return TraceSpan(
            span_id=self.span_id,
            run_id=self.run_id,
            agent_name=self.agent_name,
            model=self.model,
            started_at=self.started_at,
            ended_at=self.ended_at,
            latency_ms=self.latency_ms,
            usage=TokenUsage(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                cache_creation_input_tokens=self.cache_creation_input_tokens,
                cache_read_input_tokens=self.cache_read_input_tokens,
            ),
            cost_usd=self.cost_usd,
            stop_reason=self.stop_reason,
            error=self.error,
        )


def create_db_engine(db_url: str) -> Engine:
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    return create_engine(db_url, connect_args=connect_args)


def init_db(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)
