from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class FamilySignal(BaseModel):
    model_config = {"frozen": True}

    family: str
    direction: Direction
    confidence: float = Field(ge=0, le=100)
    skill_weight: float = Field(default=1.0, ge=0)
    as_of: date
    reasoning: str = ""

    @classmethod
    def neutral(cls, family: str, as_of: date, reasoning: str = "insufficient data") -> "FamilySignal":
        return cls(family=family, direction=Direction.NEUTRAL, confidence=50.0,
                   as_of=as_of, reasoning=reasoning)


class SignalBundle(BaseModel):
    model_config = {"frozen": True}

    ticker: str
    as_of: date
    signals: list[FamilySignal] = Field(default_factory=list)
