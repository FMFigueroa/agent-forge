from datetime import datetime

from pydantic import BaseModel, Field


class RubricScore(BaseModel):
    hook_strength: int = Field(..., ge=0, le=10)
    clarity: int = Field(..., ge=0, le=10)
    persona_fit: int = Field(..., ge=0, le=10)
    engagement_potential: int = Field(..., ge=0, le=10)

    @property
    def overall(self) -> float:
        return (
            self.hook_strength + self.clarity + self.persona_fit + self.engagement_potential
        ) / 4


class Judgment(BaseModel):
    run_id: str
    judge_model: str
    scores: RubricScore
    reasoning: str
    judged_at: datetime
