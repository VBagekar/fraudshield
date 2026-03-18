from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "creditcard.csv"
MODEL_PATH = BASE_DIR / "fraud_model.pkl"


def train_and_save_model() -> None:
    """
    Train the RandomForest model on the creditcard.csv dataset and save it to disk.
    Follows the exact steps specified:
      a. Scale Amount and Time columns using StandardScaler
      b. Handle class imbalance using class_weight='balanced'
      c. Split 80/20 train/test
      d. Train RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
      e. Print accuracy, precision, recall, F1 score
      f. Save model as fraud_model.pkl using joblib
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Make sure creditcard.csv is in the project root.")

    df = pd.read_csv(DATA_PATH)

    required_columns = {"Time", "Amount", "Class"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    feature_columns = [col for col in df.columns if col != "Class"]

    X = df[feature_columns]
    y = df["Class"]

    numeric_to_scale = ["Time", "Amount"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Scale only Time/Amount and pass the rest through unchanged.
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale_time_amount", StandardScaler(with_mean=True, with_std=True), numeric_to_scale),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    joblib.dump({"pipeline": pipeline, "feature_columns": feature_columns}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


_MODEL_CACHE: Dict[str, Any] = {}


def load_model() -> Dict[str, Any]:
    if _MODEL_CACHE:
        return _MODEL_CACHE

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run `python model.py` to train and save the model."
        )

    _MODEL_CACHE.update(joblib.load(MODEL_PATH))
    return _MODEL_CACHE


def _map_time_of_day_to_time_value(time_of_day: str) -> float:
    mapping = {
        "Morning": 9.0,
        "Afternoon": 14.0,
        "Evening": 19.0,
        "Night": 1.0,
    }
    return mapping.get(time_of_day, 12.0)


def predict_transaction(features: Dict[str, Any]) -> float:
    """
    Predict the fraud probability for a single transaction.

    The training dataset uses columns: Time, V1-V28, Amount.
    Incoming API features are business-level fields:
      - amount
      - merchant_category
      - time_of_day
      - location_risk_score

    For now, we map these into the model feature space by:
      - Using `amount` directly as Amount
      - Converting `time_of_day` into a numeric Time value
      - Filling V1-V28 with zeros (model still benefits from Amount/Time patterns)
    """
    model_bundle = load_model()
    pipeline: Pipeline = model_bundle["pipeline"]
    feature_columns = model_bundle["feature_columns"]

    amount = float(features.get("amount", 0.0))
    time_of_day = str(features.get("time_of_day", "Afternoon"))

    time_value = _map_time_of_day_to_time_value(time_of_day)

    row: Dict[str, float] = {}
    for col in feature_columns:
        if col == "Amount":
            row[col] = amount
        elif col == "Time":
            row[col] = time_value
        else:
            row[col] = 0.0

    X = pd.DataFrame([row], columns=feature_columns)
    proba = pipeline.predict_proba(X)[0][1]
    return float(proba)


if __name__ == "__main__":
    train_and_save_model()

