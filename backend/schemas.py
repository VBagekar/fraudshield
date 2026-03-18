from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_category: str = Field(..., min_length=1, max_length=100)
    time_of_day: str = Field(..., description="Human-readable time bucket e.g. Morning, Afternoon, Evening, Night")
    location_risk_score: float = Field(..., ge=0, le=10, description="Precomputed location risk score (0-10)")


class TransactionResponse(BaseModel):
    id: str
    amount: float
    merchant_category: str
    time_of_day: str
    location_risk_score: float
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    created_at: datetime

    class Config:
        from_attributes = True


class StatsResponse(BaseModel):
    total_transactions: int
    fraud_count: int
    fraud_rate: float
    avg_fraud_probability: float

