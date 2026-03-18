import datetime as dt
import uuid

from sqlalchemy import Column, String, Float, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.types import CHAR

from .database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(
        PG_UUID(as_uuid=True) if PG_UUID is not None else CHAR(36),
        primary_key=True,
        default=uuid.uuid4,
    )
    amount = Column(Float, nullable=False)
    merchant_category = Column(String(100), nullable=False)
    time_of_day = Column(String(32), nullable=False)
    location_risk_score = Column(Float, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=dt.datetime.utcnow, index=True)

