"""
Central configuration for the Livestock Health Early Warning System MVP.

Every tunable constant lives here so the risk formula, rule thresholds and
file paths are never buried inside application logic.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

REPORTS_CSV = os.path.join(DATA_DIR, "livestock_reports.csv")
REGIONS_CSV = os.path.join(DATA_DIR, "regions.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "risk_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Domain vocab
# --------------------------------------------------------------------------
SPECIES = ["cattle", "buffalo", "goat", "sheep", "poultry"]

# Symptom -> severity weight (higher = more concerning). Used to build a
# single symptom_severity feature; the model still sees each flag too.
SYMPTOM_SEVERITY = {
    "fever": 1,
    "coughing": 1,
    "nasal_discharge": 1,
    "weakness": 1,
    "diarrhea": 2,
    "loss_of_appetite": 2,
    "reduced_milk_production": 2,
    "skin_lesions": 3,
    "swelling": 3,
    "respiratory_distress": 4,
    "neurological_symptoms": 5,
    "sudden_death": 5,
}
SYMPTOM_COLUMNS = list(SYMPTOM_SEVERITY.keys())

RISK_LEVELS = ["LOW", "MEDIUM", "HIGH"]

# --------------------------------------------------------------------------
# Rule-based safety fallback thresholds
# --------------------------------------------------------------------------
RULE_MIN_DEATHS_WITH_AFFECTED = 3
RULE_MIN_AFFECTED_FOR_DEATH_RULE = 5
RULE_MORTALITY_RATE_THRESHOLD = 0.30
RULE_RESPIRATORY_MIN_AFFECTED = 3
# "rapid spread": recent reports for the same village within RAPID_SPREAD_WINDOW_DAYS
# reaching this count triggers the flag.
RAPID_SPREAD_WINDOW_DAYS = 3
RAPID_SPREAD_MIN_REPORTS = 3

# --------------------------------------------------------------------------
# Regional risk scoring formula (weights must sum to 1.0)
# --------------------------------------------------------------------------
RISK_WEIGHTS = {
    "report_volume": 0.25,
    "recent_growth": 0.20,
    "mortality": 0.20,
    "high_concern_ratio": 0.20,
    "historical_incidence": 0.15,
}
assert abs(sum(RISK_WEIGHTS.values()) - 1.0) < 1e-9, "RISK_WEIGHTS must sum to 1.0"

ROLLING_WINDOW_SHORT_DAYS = 7
ROLLING_WINDOW_LONG_DAYS = 30

# Normalization caps used to scale raw counts into a 0-100 sub-score before
# combining with the weights above. These are deliberately simple and are
# tuned to the synthetic dataset's scale - documented here, not hidden.
NORMALIZATION_CAPS = {
    "report_volume_cap": 20,      # reports in the short window considered "saturating"
    "recent_growth_cap": 3.0,     # growth ratio (short window vs prior period) considered saturating
    "mortality_rate_cap": 0.5,    # mortality rate considered saturating
    "historical_incidence_cap": 10,  # past outbreak events considered saturating
}

RISK_CATEGORY_THRESHOLDS = [
    (0, 29, "LOW"),
    (30, 59, "MEDIUM"),
    (60, 79, "HIGH"),
    (80, 100, "CRITICAL"),
]


def risk_score_to_category(score: float) -> str:
    """Map a 0-100 risk score to its category label."""
    score = max(0.0, min(100.0, score))
    if score < 30:
        return "LOW"
    if score < 60:
        return "MEDIUM"
    if score < 80:
        return "HIGH"
    return "CRITICAL"
