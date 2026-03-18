import logging
from typing import List

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from model import predict_transaction, load_model
from models import Transaction
from schemas import TransactionInput, TransactionResponse, StatsResponse


logger = logging.getLogger("fraudshield")

app = FastAPI(title="FraudShield API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    try:
        load_model()
        logger.info("Fraud detection model loaded successfully.")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load fraud detection model: %s", exc)
        raise


def _risk_level_from_probability(probability: float) -> str:
    if probability >= 0.8:
        return "HIGH"
    if probability >= 0.5:
        return "MEDIUM"
    return "LOW"


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=TransactionResponse, status_code=status.HTTP_201_CREATED)
async def predict(
    payload: TransactionInput,
    db: Session = Depends(get_db),
) -> TransactionResponse:
    try:
        fraud_probability = predict_transaction(payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error during prediction")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed") from exc

    is_fraud = fraud_probability >= 0.5
    risk_level = _risk_level_from_probability(fraud_probability)

    tx = Transaction(
        amount=payload.amount,
        merchant_category=payload.merchant_category,
        time_of_day=payload.time_of_day,
        location_risk_score=payload.location_risk_score,
        fraud_probability=fraud_probability,
        is_fraud=is_fraud,
    )

    try:
        db.add(tx)
        db.commit()
        db.refresh(tx)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Database error while saving transaction")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error") from exc

    return TransactionResponse(
        id=str(tx.id),
        amount=tx.amount,
        merchant_category=tx.merchant_category,
        time_of_day=tx.time_of_day,
        location_risk_score=tx.location_risk_score,
        fraud_probability=tx.fraud_probability,
        is_fraud=tx.is_fraud,
        risk_level=risk_level,
        created_at=tx.created_at,
    )


@app.get("/transactions", response_model=List[TransactionResponse])
async def list_transactions(db: Session = Depends(get_db)) -> List[TransactionResponse]:
    try:
        records = (
            db.query(Transaction)
            .filter(Transaction.is_fraud.is_(True))
            .order_by(Transaction.created_at.desc())
            .limit(50)
            .all()
        )
    except SQLAlchemyError as exc:
        logger.exception("Database error while fetching transactions")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error") from exc

    responses: List[TransactionResponse] = []
    for tx in records:
        responses.append(
            TransactionResponse(
                id=str(tx.id),
                amount=tx.amount,
                merchant_category=tx.merchant_category,
                time_of_day=tx.time_of_day,
                location_risk_score=tx.location_risk_score,
                fraud_probability=tx.fraud_probability,
                is_fraud=tx.is_fraud,
                risk_level=_risk_level_from_probability(tx.fraud_probability),
                created_at=tx.created_at,
            )
        )
    return responses


@app.get("/stats", response_model=StatsResponse)
async def stats(db: Session = Depends(get_db)) -> StatsResponse:
    try:
        total_transactions = db.query(func.count(Transaction.id)).scalar() or 0
        fraud_count = (
            db.query(func.count(Transaction.id)).filter(Transaction.is_fraud.is_(True)).scalar() or 0
        )
        avg_fraud_probability = (
            db.query(func.avg(Transaction.fraud_probability)).scalar() or 0.0
        )
    except SQLAlchemyError as exc:
        logger.exception("Database error while fetching stats")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error") from exc

    fraud_rate = float(fraud_count) / float(total_transactions) if total_transactions else 0.0

    return StatsResponse(
        total_transactions=int(total_transactions),
        fraud_count=int(fraud_count),
        fraud_rate=fraud_rate,
        avg_fraud_probability=float(avg_fraud_probability),
    )

