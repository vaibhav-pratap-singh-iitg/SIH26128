"""
Rule-based safety fallback engine.

This sits *alongside* the ML model, not inside it. Its job is to catch
obvious red-flag situations that must never be softened by a probabilistic
prediction - e.g. multiple sudden deaths - and force the final risk level to
HIGH regardless of what the model says. Every rule is explicit and
explainable; there is no hidden logic here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src import config


@dataclass
class RuleResult:
    triggered: bool
    rule_name: Optional[str]
    explanation: Optional[str]


def _rate(deaths: int, affected: int) -> float:
    if not affected:
        return 0.0
    return deaths / affected


def evaluate_rules(report: dict, rapid_spread: bool = False) -> RuleResult:
    """
    Evaluate the safety rule set against a single report.

    `report` is expected to contain: number_affected, number_deaths, and the
    boolean symptom flags produced by preprocessing (keys like
    "symptom_neurological_symptoms", "symptom_respiratory_distress").

    `rapid_spread` is computed upstream from regional report-frequency data
    (see aggregation.py) since a single report can't know about recent
    village-level trends on its own.
    """
    affected = int(report.get("number_affected", 0) or 0)
    deaths = int(report.get("number_deaths", 0) or 0)
    mortality_rate = _rate(deaths, affected)

    neurological = bool(report.get("symptom_neurological_symptoms", False))
    sudden_death = bool(report.get("symptom_sudden_death", False))
    respiratory = bool(report.get("symptom_respiratory_distress", False))

    # Order matters only for which explanation is surfaced first; any single
    # triggered rule is enough to force HIGH.
    if deaths >= config.RULE_MIN_DEATHS_WITH_AFFECTED and affected >= config.RULE_MIN_AFFECTED_FOR_DEATH_RULE:
        return RuleResult(
            True, "multiple_deaths",
            f"{deaths} deaths reported out of {affected} affected animals - "
            f"meets the multiple-mortality emergency threshold.",
        )

    if mortality_rate >= config.RULE_MORTALITY_RATE_THRESHOLD:
        return RuleResult(
            True, "high_mortality_rate",
            f"Mortality rate of {mortality_rate:.0%} exceeded the configured "
            f"emergency threshold of {config.RULE_MORTALITY_RATE_THRESHOLD:.0%}.",
        )

    if neurological or sudden_death:
        return RuleResult(
            True, "critical_symptoms",
            "Neurological symptoms or sudden death reported - treated as an "
            "automatic high-concern signal regardless of other factors.",
        )

    if respiratory and affected >= config.RULE_RESPIRATORY_MIN_AFFECTED:
        return RuleResult(
            True, "respiratory_cluster",
            f"Respiratory distress reported across {affected} animals, at or "
            f"above the {config.RULE_RESPIRATORY_MIN_AFFECTED}-animal cluster threshold.",
        )

    if rapid_spread:
        return RuleResult(
            True, "rapid_spread",
            f"{config.RAPID_SPREAD_MIN_REPORTS}+ reports from this village within "
            f"{config.RAPID_SPREAD_WINDOW_DAYS} days - flagged as a rapidly "
            f"spreading situation.",
        )

    return RuleResult(False, None, None)
