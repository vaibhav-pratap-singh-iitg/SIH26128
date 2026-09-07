"""
Train the triage risk-level classifier.

Model: RandomForestClassifier - chosen over gradient boosting for this MVP
because it trains fast, is robust to the modest feature count here, and
`class_weight="balanced"` gives a simple, well-understood way to push recall
on the minority HIGH class without hand-tuning a threshold. Not a deep
learning model, by design - the dataset and problem don't call for one, and
an interpretable model is easier to defend to judges and to override with
the rule engine.

Run:
    python src/train_model.py
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, preprocessing


def load_engineered_data() -> pd.DataFrame:
    df = pd.read_csv(config.REPORTS_CSV)
    return preprocessing.engineer_features(df, add_context=True)


def train(save: bool = True) -> dict:
    df = load_engineered_data()
    X = preprocessing.to_feature_frame(df)
    y = df["risk_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y,
    )

    preprocessor = preprocessing.build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train_t, y_train)

    y_pred = model.predict(X_test_t)

    labels = config.RISK_LEVELS
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
    high_recall = recall_score(y_test, y_pred, labels=["HIGH"], average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report_text = classification_report(y_test, y_pred, labels=labels, zero_division=0)

    print("=" * 60)
    print("TRIAGE MODEL EVALUATION")
    print("=" * 60)
    print(f"Accuracy:            {acc:.3f}")
    print(f"Macro precision:     {precision:.3f}")
    print(f"Macro recall:        {recall:.3f}")
    print(f"Macro F1:            {f1:.3f}")
    print(f"HIGH-concern recall: {high_recall:.3f}  <-- the metric that matters most here")
    print()
    print("Confusion matrix (rows=actual, cols=predicted), order:", labels)
    print(cm)
    print()
    print(report_text)

    if hasattr(model, "feature_importances_"):
        feature_names = preprocessor.get_feature_names_out()
        importances = pd.Series(model.feature_importances_, index=feature_names)
        print("Top 10 most influential features:")
        print(importances.sort_values(ascending=False).head(10).to_string())

    if save:
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        joblib.dump(model, config.MODEL_PATH)
        joblib.dump(preprocessor, config.PREPROCESSOR_PATH)
        print(f"\nSaved model -> {config.MODEL_PATH}")
        print(f"Saved preprocessor -> {config.PREPROCESSOR_PATH}")

    return {
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
        "high_recall": high_recall, "confusion_matrix": cm.tolist(),
    }


if __name__ == "__main__":
    train()
