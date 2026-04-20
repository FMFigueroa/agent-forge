from datetime import datetime

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


class ModelPricing(BaseModel):
    input_per_mtok: float
    output_per_mtok: float


PRICING: dict[str, ModelPricing] = {
    "claude-opus-4-7": ModelPricing(input_per_mtok=5.00, output_per_mtok=25.00),
    "claude-opus-4-6": ModelPricing(input_per_mtok=5.00, output_per_mtok=25.00),
    "claude-sonnet-4-6": ModelPricing(input_per_mtok=3.00, output_per_mtok=15.00),
    "claude-haiku-4-5": ModelPricing(input_per_mtok=1.00, output_per_mtok=5.00),
    "claude-haiku-4-5-20251001": ModelPricing(input_per_mtok=1.00, output_per_mtok=5.00),
}

CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1


class TraceSpan(BaseModel):
    span_id: str
    run_id: str
    agent_name: str
    model: str
    started_at: datetime
    ended_at: datetime
    latency_ms: float
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    stop_reason: str | None = None
    error: str | None = None


class AgentResult(BaseModel):
    text: str
    span: TraceSpan
