import time
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from anthropic import AsyncAnthropic

from agent_forge.agents.models import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    PRICING,
    AgentResult,
    TokenUsage,
    TraceSpan,
)
from agent_forge.observability.tracer import Tracer


def compute_cost(model: str, usage: TokenUsage) -> float:
    pricing = PRICING.get(model)
    if pricing is None:
        return 0.0

    per_input_tok = pricing.input_per_mtok / 1_000_000
    per_output_tok = pricing.output_per_mtok / 1_000_000

    return (
        usage.input_tokens * per_input_tok
        + usage.output_tokens * per_output_tok
        + usage.cache_creation_input_tokens * per_input_tok * CACHE_WRITE_MULTIPLIER
        + usage.cache_read_input_tokens * per_input_tok * CACHE_READ_MULTIPLIER
    )


class Agent(ABC):
    def __init__(
        self,
        *,
        name: str,
        model: str,
        client: AsyncAnthropic,
        tracer: Tracer,
        max_tokens: int = 16000,
    ) -> None:
        self.name = name
        self.model = model
        self.client = client
        self.tracer = tracer
        self.max_tokens = max_tokens

    @property
    @abstractmethod
    def system_prompt(self) -> str: ...

    async def run(self, user_message: str, *, run_id: str | None = None) -> AgentResult:
        run_id = run_id or str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        started_at = datetime.now(UTC)
        t0 = time.perf_counter()

        try:
            response: Any = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:
            self._record(
                span_id=span_id,
                run_id=run_id,
                started_at=started_at,
                latency_ms=(time.perf_counter() - t0) * 1000,
                usage=TokenUsage(),
                stop_reason=None,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        latency_ms = (time.perf_counter() - t0) * 1000
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_input_tokens=getattr(response.usage, "cache_creation_input_tokens", 0)
            or 0,
            cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        )
        text = "".join(b.text for b in response.content if b.type == "text")

        span = self._record(
            span_id=span_id,
            run_id=run_id,
            started_at=started_at,
            latency_ms=latency_ms,
            usage=usage,
            stop_reason=response.stop_reason,
            error=None,
        )
        return AgentResult(text=text, span=span)

    def _record(
        self,
        *,
        span_id: str,
        run_id: str,
        started_at: datetime,
        latency_ms: float,
        usage: TokenUsage,
        stop_reason: str | None,
        error: str | None,
    ) -> TraceSpan:
        span = TraceSpan(
            span_id=span_id,
            run_id=run_id,
            agent_name=self.name,
            model=self.model,
            started_at=started_at,
            ended_at=datetime.now(UTC),
            latency_ms=latency_ms,
            usage=usage,
            cost_usd=compute_cost(self.model, usage),
            stop_reason=stop_reason,
            error=error,
        )
        self.tracer.record(span)
        return span
