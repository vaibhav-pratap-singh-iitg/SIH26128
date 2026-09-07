"""
Feature engineering for the triage model.

Turns a raw report (plus its village's recent history) into the numeric
feature vector the ML model was trained on. The same feature list is used
at training time (train_model.py, over the whole historical dataset) and at
prediction time (predictor.py, for a single new report), so the two must
always stay in sync - that's why both go through `engineer_features` /
`FEATURE_COLUMNS` defined here rather than duplicating logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config

BOOLEAN_FEATURES = [f"symptom_{s}" for s in config.SYMPTOM_COLUMNS]
NUMERIC_FEATURES = [
    "number_affected", "number_deaths", "mortality_rate", "duration_days",
    "temperature", "symptom_count", "symptom_severity",
    "recent_reports", "historical_incidence", "spread_velocity",
]
CATEGORICAL_FEATURES = ["species"]
FEATURE_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES


def _symptom_derived(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    severity_weights = config.SYMPTOM_SEVERITY
    d["symptom_count"] = d[BOOLEAN_FEATURES].sum(axis=1)
    d["symptom_severity"] = sum(
        d[f"symptom_{s}"].astype(int) * w for s, w in severity_weights.items()
    )
    return d


def add_recent_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every report, compute village-level context using ONLY reports
    strictly before it (or, for the same-day recent window, excluding
    itself) - i.e. no leakage of a report's own outcome into its own
    features. Vectorized per-village for speed.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["village", "date"]).reset_index(drop=True)

    has_labels = "risk_level" in d.columns
    recent_reports = np.zeros(len(d), dtype=int)
    historical_incidence = np.zeros(len(d), dtype=int)
    spread_velocity = np.zeros(len(d), dtype=float)
    rapid_spread = np.zeros(len(d), dtype=bool)

    short_win = np.timedelta64(config.ROLLING_WINDOW_SHORT_DAYS - 1, "D")
    prev_win = np.timedelta64(config.ROLLING_WINDOW_SHORT_DAYS, "D")
    rapid_win = np.timedelta64(config.RAPID_SPREAD_WINDOW_DAYS - 1, "D")
    one_day = np.timedelta64(1, "D")

    for _, idx in d.groupby("village", sort=False).groups.items():
        idx = list(idx)
        dates = d.loc[idx, "date"].values.astype("datetime64[D]")
        is_high = d.loc[idx, "risk_level"].values == "HIGH" if has_labels else np.zeros(len(idx), dtype=bool)
        n = len(idx)
        for i in range(n):
            as_of = dates[i]
            short_start = as_of - short_win
            prev_start = short_start - prev_win
            prev_end = short_start - one_day
            rapid_start = as_of - rapid_win

            recent_mask = (dates >= short_start) & (dates <= as_of)
            prev_mask = (dates >= prev_start) & (dates <= prev_end)
            hist_mask = dates < as_of
            rapid_mask = (dates >= rapid_start) & (dates <= as_of)

            recent_count = int(recent_mask.sum()) - 1  # exclude self
            prev_count = int(prev_mask.sum())
            hist_count = int(is_high[hist_mask].sum())
            rapid_count = int(rapid_mask.sum()) - 1  # exclude self

            spread_vel = (recent_count / prev_count) if prev_count else float(recent_count)

            row_pos = idx[i]
            recent_reports[row_pos] = max(0, recent_count)
            historical_incidence[row_pos] = hist_count
            spread_velocity[row_pos] = round(float(spread_vel), 3)
            rapid_spread[row_pos] = rapid_count >= config.RAPID_SPREAD_MIN_REPORTS

    d["recent_reports"] = recent_reports
    d["historical_incidence"] = historical_incidence
    d["spread_velocity"] = spread_velocity
    d["rapid_spread"] = rapid_spread
    return d


def engineer_features(df: pd.DataFrame, add_context: bool = True) -> pd.DataFrame:
    """Full feature engineering pass used at both training and prediction time."""
    d = df.copy()
    if "mortality_rate" not in d.columns:
        d["mortality_rate"] = (d["number_deaths"] / d["number_affected"].replace(0, np.nan)).fillna(0.0)
    d = _symptom_derived(d)
    if add_context:
        d = add_recent_context_features(d)
    return d


def build_preprocessor() -> ColumnTransformer:
    """
    ColumnTransformer: median-impute numeric/boolean columns (temperature is
    optional and will have missing values), one-hot encode species. No
    scaling - the model is tree-based and scale-invariant, so scaling would
    add complexity without changing predictions.
    """
    numeric_and_boolean = NUMERIC_FEATURES + BOOLEAN_FEATURES
    numeric_pipeline = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_and_boolean),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def to_feature_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Select and order exactly the columns the preprocessor expects."""
    out = d.copy()
    for col in BOOLEAN_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype(int)
    missing = [c for c in FEATURE_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")
    return out[FEATURE_COLUMNS]
