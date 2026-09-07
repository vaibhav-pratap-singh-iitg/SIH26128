"""
Transparent regional risk scoring.

Regional Risk Score (0-100) =
      25% report volume        (reports in the last 7 days)
    + 20% recent growth        (7-day count vs the prior 7-day count)
    + 20% mortality            (mortality rate in the last 7 days)
    + 20% high-concern ratio   (share of last-7-day reports the model/rules called HIGH)
    + 15% historical incidence (past HIGH-concern reports for this region)

Every component is normalized to 0-100 against a documented cap (see
config.NORMALIZATION_CAPS) before the weights are applied, so no single
saturating input can blow up the score. Weights and caps live in config.py,
not here, so they can be tuned in one place.
"""

from __future__ import annotations

import pandas as pd

from src import aggregation, config


def _normalize(value: float, cap: float) -> float:
    """Scale a raw value to 0-100 against a cap, clipped at both ends."""
    if cap <= 0:
        return 0.0
    return max(0.0, min(100.0, (value / cap) * 100.0))


def score_row(row: pd.Series) -> dict:
    """Compute the 0-100 risk score (and its sub-scores) for one aggregated region row."""
    caps = config.NORMALIZATION_CAPS
    w = config.RISK_WEIGHTS

    volume_score = _normalize(row["report_count_7d"], caps["report_volume_cap"])
    growth_score = _normalize(max(0.0, row["report_growth_rate"] - 1.0), caps["recent_growth_cap"])
    mortality_score = _normalize(row["mortality_rate"], caps["mortality_rate_cap"])
    high_concern_score = _normalize(row["high_concern_ratio"], 1.0)  # ratio is already 0-1
    historical_score = _normalize(row["historical_incidence"], caps["historical_incidence_cap"])

    total = (
        w["report_volume"] * volume_score
        + w["recent_growth"] * growth_score
        + w["mortality"] * mortality_score
        + w["high_concern_ratio"] * high_concern_score
        + w["historical_incidence"] * historical_score
    )
    total = round(max(0.0, min(100.0, total)), 1)

    return {
        "risk_score": total,
        "risk_category": config.risk_score_to_category(total),
        "sub_score_volume": round(volume_score, 1),
        "sub_score_growth": round(growth_score, 1),
        "sub_score_mortality": round(mortality_score, 1),
        "sub_score_high_concern": round(high_concern_score, 1),
        "sub_score_historical": round(historical_score, 1),
    }


def _calculate_risk(df: pd.DataFrame, region_col: str, as_of_date=None) -> pd.DataFrame:
    rollup = aggregation.aggregate_regions(df, region_col, as_of_date=as_of_date)
    if rollup.empty:
        return rollup
    scored = rollup.apply(score_row, axis=1, result_type="expand")
    return pd.concat([rollup, scored], axis=1).sort_values("risk_score", ascending=False).reset_index(drop=True)


def calculate_village_risk(df: pd.DataFrame, as_of_date=None) -> pd.DataFrame:
    return _calculate_risk(df, "village", as_of_date=as_of_date)


def calculate_block_risk(df: pd.DataFrame, as_of_date=None) -> pd.DataFrame:
    return _calculate_risk(df, "block", as_of_date=as_of_date)


def calculate_district_risk(df: pd.DataFrame, as_of_date=None) -> pd.DataFrame:
    return _calculate_risk(df, "district", as_of_date=as_of_date)


def explain_region_risk(row: pd.Series) -> list[str]:
    """Human-readable bullet points explaining what drove a region's score."""
    reasons = []
    if row.get("sub_score_mortality", 0) >= 40:
        reasons.append(f"Elevated mortality rate ({row['mortality_rate']:.0%} over the last 7 days)")
    if row.get("sub_score_growth", 0) >= 30:
        reasons.append(f"Reports growing quickly (7-day growth ratio {row['report_growth_rate']:.2f}x)")
    if row.get("sub_score_volume", 0) >= 40:
        reasons.append(f"High report volume ({int(row['report_count_7d'])} reports in 7 days)")
    if row.get("sub_score_high_concern", 0) >= 30:
        reasons.append(f"Large share of HIGH-concern reports ({row['high_concern_ratio']:.0%})")
    if row.get("sub_score_historical", 0) >= 30:
        reasons.append(f"History of past outbreaks ({int(row['historical_incidence'])} prior HIGH reports)")
    if not reasons:
        reasons.append("No single factor dominates - risk is low across all components.")
    return reasons
