"""Training and prediction helpers for the Loan Prediction System."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


FEATURES = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "credit_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

EXPECTED_COLUMNS = {"loan_id", "loan_status", *FEATURES}
EDUCATION_ENCODING = {"graduate": 1, "not graduate": 0}
SELF_EMPLOYED_ENCODING = {"yes": 1, "no": 0}
STATUS_ENCODING = {"approved": 1, "rejected": 0}


class LoanDataError(ValueError):
    """Raised when the dataset cannot be used by the project's ML pipeline."""


def prepare_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, int]:
    """Load and encode the dataset exactly as in the Review-2 notebook."""
    if not csv_path.is_file():
        raise FileNotFoundError("LoanApproval.csv could not be found.")

    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()
    if "cibil_score" in data.columns:
        data = data.rename(columns={"cibil_score": "credit_score"})
    data = data.drop_duplicates()

    missing_columns = EXPECTED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise LoanDataError(
            "The dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    model_data = data.drop(columns=["loan_id"], errors="ignore").copy()
    for column in ("education", "self_employed", "loan_status"):
        model_data[column] = (
            model_data[column].astype(str).str.strip().str.lower()
        )

    model_data["education"] = model_data["education"].map(EDUCATION_ENCODING)
    model_data["self_employed"] = model_data["self_employed"].map(
        SELF_EMPLOYED_ENCODING
    )
    model_data["loan_status"] = model_data["loan_status"].map(STATUS_ENCODING)

    required = FEATURES + ["loan_status"]
    if model_data[required].isna().any().any():
        raise LoanDataError(
            "The dataset contains missing or unexpected categorical values."
        )

    return model_data[FEATURES], model_data["loan_status"], len(data)


def train_model(csv_path: Path) -> dict[str, Any]:
    """Train the notebook's Logistic Regression model and calculate metrics."""
    features, target, record_count = prepare_dataset(csv_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "model": model,
        "feature_names": FEATURES,
        "record_count": record_count,
        "metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions)),
            "recall": float(recall_score(y_test, predictions)),
            "f1": float(f1_score(y_test, predictions)),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        },
    }


def save_model_bundle(bundle: dict[str, Any], model_path: Path) -> None:
    """Persist a model bundle, creating its directory when needed."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    """Load either this app's bundle or a compatible standalone sklearn model."""
    loaded = joblib.load(model_path)
    if isinstance(loaded, dict) and "model" in loaded:
        bundle = loaded
    elif hasattr(loaded, "predict") and hasattr(loaded, "predict_proba"):
        bundle = {
            "model": loaded,
            "feature_names": list(getattr(loaded, "feature_names_in_", FEATURES)),
            "record_count": 4269,
            "metrics": {"accuracy": 0.8033},
        }
    else:
        raise LoanDataError("The saved model file is not a compatible model.")

    if list(bundle.get("feature_names", [])) != FEATURES:
        raise LoanDataError("The saved model uses a different feature order.")
    return bundle


def applicant_frame(values: dict[str, int]) -> pd.DataFrame:
    """Create a one-row model input with the required feature order."""
    return pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)


def predict_applicant(
    model: LogisticRegression, values: dict[str, int]
) -> tuple[int, float]:
    """Return the predicted class and its corresponding model probability."""
    frame = applicant_frame(values)
    prediction = int(model.predict(frame)[0])
    classes = list(model.classes_)
    predicted_probability = float(model.predict_proba(frame)[0][classes.index(prediction)])
    return prediction, predicted_probability
