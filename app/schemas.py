"""Pydantic request/response schemas."""
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(..., description="Text to analyse", max_length=5000,
                      examples=["This product is excellent and I love it."])

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Text must not be empty.")
        return value


class PredictResponse(BaseModel):
    text: str
    sentiment: str = Field(..., description="'positive' or 'negative'")
    confidence: float = Field(..., ge=0, le=1)
    probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    detail: str


class RootResponse(BaseModel):
    name: str
    version: str
    description: str
    endpoints: List[str]
