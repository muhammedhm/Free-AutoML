"""Request bodies (validated by pydantic)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ExperimentRequest(BaseModel):
    dataset_id: str
    target: str
    task: Literal["classification", "regression"] | None = None   # None = auto-detect
    metric: str | None = None                                      # None = default for the task
    time_budget: int = Field(60, ge=10, le=1800, description="Search time in seconds")
    test_size: float = Field(0.2, ge=0.1, le=0.5)
    estimators: list[str] | None = None
    name: str | None = None


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1, max_length=10_000)
    with_proba: bool = False
