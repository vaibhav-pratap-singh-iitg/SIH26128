"""
End-to-end prediction pipeline for a single incoming report.

Pipeline: raw report -> feature engineering -> ML prediction -> rule-based
safety fallback -> final risk level + plain-language explanation.

This module is deliberately explicit about the difference between the ML
model's opinion and the rule engine's override, because that distinction is
what makes the system explainable and safe to demo to judges (and, in a real
deployment, safe to trust).
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src import config, preprocessing, rules

RECOMMENDED_ACTION = {
    "LOW": "Log for routine monitoring. No immediate action required.",
    "MEDIUM": "Notify the local veterinary worker for a follow-up visit within a few days.",
    "HIGH": "Escalate for urgent veterinary assessment and consider sample collection.",
}


class ModelNotTrainedError(RuntimeError):
    pass


# Manual caching instead of @lru_cache — lru_cache permanently caches
# ModelNotTrainedError on Colab if the app starts before training finishes,
# and even re-running won't fix it.  Manual caching allows retry.
_cached_artifacts = None


def _load_artifacts():
    global _cached_artifacts
    if _cached_artifacts is not None:
        return _cached_artifacts
    try:
        model = joblib.load(config.MODEL_PATH)
        preprocessor = joblib.load(config.PREPROCESSOR_PATH)
    except FileNotFoundError as exc:
        raise ModelNotTrainedError(
            "Model artifacts not found. Run `python src/train_model.py` first."
        ) from exc
    _cached_artifacts = (model, preprocessor)
    return _cached_artifacts


def _validate_report(report: dict) -> list[str]:
    errors = []
    if report.get("number_affected") is None or int(report.get("number_affected", 0)) <= 0:
        errors.append("Number of affected animals must be a positive integer.")
    affected = int(report.get("number_affected") or 0)
    deaths = report.get("number_deaths")
    if deaths is None or int(deaths) < 0:
        errors.append("Number of deaths cannot be negative.")
    elif affected and int(deaths) > affected:
        errors.append("Number of deaths cannot exceed the number of affected animals.")
    if report.get("species") not in config.SPECIES:
        errors.append(f"Species must be one of: {', '.join(config.SPECIES)}.")
    if not report.get("district") or not report.get("block") or not report.get("village"):
        errors.append("District, block and village are all required.")
    if report.get("date") is None:
        errors.append("Report date is required.")
    return errors


def predict_report(report: dict, history_df: pd.DataFrame) -> dict:
    """
    Run the full pipeline for one new report.

    `report` should contain: date, district, block, village, species,
    number_affected, number_deaths, duration_days, temperature (optional),
    notes (optional), and symptom_<name> booleans for each of
    config.SYMPTOM_COLUMNS.

    `history_df` is the existing reports dataset, used to compute this
    village's recent-report context (recent_reports, historical_incidence,
    spread_velocity, rapid_spread).
    """
    errors = _validate_report(report)
    if errors:
        return {"valid": False, "errors": errors}

    model, preprocessor = _load_artifacts()

    row = dict(report)
    row.setdefault("number_deaths", 0)
    row.setdefault("duration_days", 1)
    row.setdefault("temperature", np.nan)
    for s in config.SYMPTOM_COLUMNS:
        row.setdefault(f"symptom_{s}", False)

    single_df = pd.DataFrame([row])
    single_df["mortality_rate"] = (
        single_df["number_deaths"] / single_df["number_affected"].replace(0, np.nan)
    ).fillna(0.0)
    engineered = preprocessing.engineer_features(single_df, add_context=False)

    context = _village_context(history_df, row["village"], row["date"])
    for key in ("recent_reports", "historical_incidence", "spread_velocity"):
        engineered[key] = context[key]

    features = preprocessing.to_feature_frame(engineered)
    features_t = preprocessor.transform(features)

    proba = model.predict_proba(features_t)[0]
    class_labels = list(model.classes_)
    ml_prediction = class_labels[int(np.argmax(proba))]
    ml_confidence = float(np.max(proba))
    proba_by_class = {cls: round(float(p), 3) for cls, p in zip(class_labels, proba)}

    rule_result = rules.evaluate_rules(row, rapid_spread=context["rapid_spread"])

    if rule_result.triggered:
        final_risk = "HIGH"
    else:
        final_risk = ml_prediction

    explanation = _build_explanation(row, ml_prediction, ml_confidence, rule_result, context)

    return {
        "valid": True,
        "ml_prediction": ml_prediction,
        "ml_confidence": round(ml_confidence, 3),
        "ml_probabilities": proba_by_class,
        "rule_triggered": rule_result.triggered,
        "rule_name": rule_result.rule_name,
        "rule_explanation": rule_result.explanation,
        "final_risk": final_risk,
        "recommended_action": RECOMMENDED_ACTION[final_risk],
        "explanation": explanation,
        "context": context,
    }


def _village_context(history_df: pd.DataFrame, village: str, date) -> dict:
    from src import aggregation
    if history_df is None or history_df.empty:
        return {
            "recent_reports": 0, "recent_reports_prev": 0,
            "historical_incidence": 0, "spread_velocity": 0.0, "rapid_spread": False,
        }
    return aggregation.compute_recent_context(history_df, village, date)


def _build_explanation(row: dict, ml_prediction: str, ml_confidence: float,
                        rule_result: rules.RuleResult, context: dict) -> list[str]:
    """Plain-language bullet points on what drove the final risk level."""
    reasons = []
    affected = int(row.get("number_affected", 0) or 0)
    deaths = int(row.get("number_deaths", 0) or 0)
    mortality_rate = (deaths / affected) if affected else 0.0

    if mortality_rate > 0:
        reasons.append(f"Mortality rate for this report: {mortality_rate:.0%} ({deaths} of {affected} animals)")
    active_symptoms = [s for s in config.SYMPTOM_COLUMNS if row.get(f"symptom_{s}")]
    if active_symptoms:
        reasons.append("Symptoms reported: " + ", ".join(s.replace("_", " ") for s in active_symptoms))
    if context["recent_reports"] > 0:
        reasons.append(f"{context['recent_reports']} other report(s) from this village in the last "
                        f"{config.ROLLING_WINDOW_SHORT_DAYS} days")
    if context["historical_incidence"] > 0:
        reasons.append(f"{context['historical_incidence']} past HIGH-concern report(s) on record for this village")
    reasons.append(f"ML model classified this report as {ml_prediction} "
                    f"(confidence {ml_confidence:.0%}) based on the factors above")
    if rule_result.triggered:
        reasons.append(f"Safety rule override triggered: {rule_result.explanation}")
    return reasons
