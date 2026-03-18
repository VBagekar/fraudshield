"""
FraudShield — model.py
Trains a Random Forest on the 4 features the frontend actually sends:
  - amount
  - merchant_category (encoded)
  - time_of_day (encoded)
  - location_risk_score

Generates 50,000 synthetic transactions with realistic fraud patterns,
trains, evaluates, and saves the model as fraud_model.pkl
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fraudshield.model")

BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fraud_model.pkl"

# ── Feature definitions ───────────────────────────────────────────
CATEGORIES = ["Online", "Retail", "Restaurant", "Travel",
              "Entertainment", "Gas", "Healthcare"]
TIMES      = ["Morning", "Afternoon", "Evening", "Night"]

# Fraud risk weights per category (higher = more fraudulent)
CATEGORY_RISK = {
    "Online":        0.35,
    "Travel":        0.25,
    "Entertainment": 0.20,
    "Retail":        0.10,
    "Gas":           0.08,
    "Restaurant":    0.05,
    "Healthcare":    0.03,
}

# Fraud risk weights per time of day
TIME_RISK = {
    "Night":     0.45,
    "Evening":   0.25,
    "Morning":   0.15,
    "Afternoon": 0.10,
}

# ── Synthetic data generation ─────────────────────────────────────
def generate_dataset(n_samples: int = 50_000, fraud_rate: float = 0.08, seed: int = 42):
    """
    Generate realistic synthetic transaction data.

    Fraud patterns encoded:
      - High amount (> ₹8,000) + Night + Online/Travel → very high fraud
      - High location risk (> 7) + Night → elevated fraud
      - Low amount + Daytime + Healthcare/Restaurant → very low fraud
      - Random noise added so model can't perfectly memorise rules
    """
    rng = np.random.default_rng(seed)
    n_fraud  = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud

    # ── Normal transactions ───────────────────────────────────────
    normal_amounts    = rng.lognormal(mean=6.5, sigma=1.2, size=n_normal)   # ₹50–₹5,000 range
    normal_categories = rng.choice(CATEGORIES, size=n_normal,
                                   p=[0.25, 0.20, 0.18, 0.10, 0.10, 0.10, 0.07])
    normal_times      = rng.choice(TIMES, size=n_normal,
                                   p=[0.30, 0.35, 0.25, 0.10])
    normal_risk       = rng.uniform(0, 5, size=n_normal)                    # low risk scores

    # ── Fraudulent transactions ───────────────────────────────────
    # Mix of fraud patterns
    fraud_amounts    = np.concatenate([
        rng.uniform(8_000, 50_000, size=int(n_fraud * 0.5)),   # large amounts
        rng.uniform(1, 50,         size=int(n_fraud * 0.3)),   # tiny amounts (card testing)
        rng.uniform(500, 8_000,    size=int(n_fraud * 0.2)),   # mid amounts
    ])
    fraud_categories = rng.choice(CATEGORIES, size=n_fraud,
                                   p=[0.40, 0.20, 0.15, 0.10, 0.08, 0.04, 0.03])
    fraud_times      = rng.choice(TIMES, size=n_fraud,
                                   p=[0.10, 0.10, 0.25, 0.55])   # mostly night
    fraud_risk       = rng.uniform(5, 10, size=n_fraud)            # high risk scores

    # ── Combine ───────────────────────────────────────────────────
    amounts    = np.concatenate([normal_amounts, fraud_amounts])
    categories = np.concatenate([normal_categories, fraud_categories])
    times      = np.concatenate([normal_times, fraud_times])
    risks      = np.concatenate([normal_risk, fraud_risk])
    labels     = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])

    # Shuffle
    idx = rng.permutation(n_samples)
    df  = pd.DataFrame({
        "amount":               amounts[idx],
        "merchant_category":    categories[idx],
        "time_of_day":          times[idx],
        "location_risk_score":  risks[idx],
        "is_fraud":             labels[idx].astype(int),
    })

    # Add noise: flip ~1% of labels to prevent overfitting
    noise_idx = rng.choice(n_samples, size=int(n_samples * 0.01), replace=False)
    df.loc[noise_idx, "is_fraud"] = 1 - df.loc[noise_idx, "is_fraud"]

    return df


# ── Model training ─────────────────────────────────────────────────
def train_model():
    logger.info("Generating synthetic training data (50,000 transactions)...")
    df = generate_dataset(n_samples=50_000, fraud_rate=0.08)

    fraud_count  = df["is_fraud"].sum()
    normal_count = len(df) - fraud_count
    logger.info(f"Dataset: {normal_count:,} normal | {fraud_count:,} fraud ({fraud_count/len(df)*100:.1f}%)")

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Preprocessing pipeline ────────────────────────────────────
    numeric_features     = ["amount", "location_risk_score"]
    categorical_features = ["merchant_category", "time_of_day"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ])

    # ── Classifier ────────────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   clf),
    ])

    logger.info("Training Random Forest (200 trees)...")
    pipeline.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────
    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    auc       = roc_auc_score(y_test, y_pred_prob)

    logger.info("=" * 50)
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  ROC-AUC:   {auc:.4f}")
    logger.info("=" * 50)
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    # ── Save ──────────────────────────────────────────────────────
    artifact = {
        "pipeline":           pipeline,
        "feature_columns":    numeric_features + categorical_features,
        "numeric_features":   numeric_features,
        "categorical_features": categorical_features,
        "metrics": {
            "accuracy":  round(accuracy,  4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "auc":       round(auc,       4),
        },
    }
    joblib.dump(artifact, MODEL_PATH)
    logger.info(f"Model saved → {MODEL_PATH}")
    return pipeline


# ── Model loading (cached) ────────────────────────────────────────
_MODEL_CACHE: dict = {}

def load_model():
    if "pipeline" not in _MODEL_CACHE:
        if not MODEL_PATH.exists():
            logger.warning("Model not found — training now...")
            train_model()
        artifact = joblib.load(MODEL_PATH)
        _MODEL_CACHE.update(artifact)
        m = artifact.get("metrics", {})
        logger.info(
            f"Model loaded | AUC={m.get('auc','?')} "
            f"Recall={m.get('recall','?')} F1={m.get('f1','?')}"
        )
    return _MODEL_CACHE["pipeline"]


# ── Inference ─────────────────────────────────────────────────────
def predict_transaction(features: dict) -> float:
    """
    Returns fraud probability (0.0 – 1.0) for a single transaction.

    Expected keys:
        amount               (float)  — transaction amount in ₹
        merchant_category    (str)    — e.g. "Online", "Retail"
        time_of_day          (str)    — "Morning" / "Afternoon" / "Evening" / "Night"
        location_risk_score  (float)  — 0 to 10
    """
    pipeline = load_model()

    row = {
        "amount":               float(features.get("amount", 0.0)),
        "merchant_category":    str(features.get("merchant_category", "Online")),
        "time_of_day":          str(features.get("time_of_day", "Afternoon")),
        "location_risk_score":  float(features.get("location_risk_score", 5.0)),
    }

    X    = pd.DataFrame([row])
    prob = pipeline.predict_proba(X)[0][1]
    return round(float(prob), 4)


# ── Run training directly ─────────────────────────────────────────
if __name__ == "__main__":
    # Delete old model first so we get a clean retrain
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
        logger.info("Deleted old fraud_model.pkl")

    train_model()
    logger.info("Done. Test predictions:")

    test_cases = [
        {"amount": 45000, "merchant_category": "Online",      "time_of_day": "Night",     "location_risk_score": 9.0},
        {"amount": 150,   "merchant_category": "Restaurant",  "time_of_day": "Afternoon", "location_risk_score": 1.5},
        {"amount": 12000, "merchant_category": "Travel",      "time_of_day": "Night",     "location_risk_score": 8.0},
        {"amount": 50,    "merchant_category": "Healthcare",  "time_of_day": "Morning",   "location_risk_score": 2.0},
        {"amount": 800,   "merchant_category": "Online",      "time_of_day": "Night",     "location_risk_score": 7.5},
    ]

    for t in test_cases:
        prob = predict_transaction(t)
        tag  = "FRAUD" if prob > 0.7 else "SUSPICIOUS" if prob > 0.3 else "SAFE"
        logger.info(
            f"  ₹{t['amount']:>8,.0f} | {t['merchant_category']:<13} | "
            f"{t['time_of_day']:<10} | Risk {t['location_risk_score']} → "
            f"{prob:.4f} [{tag}]"
        )