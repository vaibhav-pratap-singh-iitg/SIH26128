"""
Aggregation utilities: rolling up individual reports into region-level and
per-village-context statistics.

Two things live here:

1. `compute_recent_context` - per-report context (used as ML features and by
   the rapid-spread rule) describing what has recently happened in *this*
   report's own village, computed only from reports strictly before it so
   there is no label leakage.

2. `aggregate_regions` - a full rollup table (one row per village/block/
   district) used by risk_scoring.py and the dashboard/map.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src import config


def _to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
    return df


def compute_recent_context(
    df: pd.DataFrame,
    village: str,
    as_of_date,
    exclude_report_id: Optional[int] = None,
) -> dict:
    """
    Village-level context for a single report: how many reports have recently
    come from this village, how that compares to the period before, and how
    many historical HIGH-concern reports this village has had.
    """
    d = df[df["village"] == village]
    if exclude_report_id is not None and "report_id" in d.columns:
        d = d[d["report_id"] != exclude_report_id]
    d = _to_datetime(d)
    as_of = pd.to_datetime(as_of_date)

    short_start = as_of - pd.Timedelta(days=config.ROLLING_WINDOW_SHORT_DAYS - 1)
    prev_start = short_start - pd.Timedelta(days=config.ROLLING_WINDOW_SHORT_DAYS)
    prev_end = short_start - pd.Timedelta(days=1)
    rapid_start = as_of - pd.Timedelta(days=config.RAPID_SPREAD_WINDOW_DAYS - 1)

    recent_mask = (d["date"] >= short_start) & (d["date"] <= as_of)
    prev_mask = (d["date"] >= prev_start) & (d["date"] <= prev_end)
    hist_mask = d["date"] < as_of
    rapid_mask = (d["date"] >= rapid_start) & (d["date"] <= as_of)

    recent_reports = int(recent_mask.sum())
    prev_reports = int(prev_mask.sum())
    historical_incidence = int((d.loc[hist_mask, "risk_level"] == "HIGH").sum()) if "risk_level" in d.columns else 0
    spread_velocity = (recent_reports / prev_reports) if prev_reports else float(recent_reports)
    rapid_spread = int(rapid_mask.sum()) >= config.RAPID_SPREAD_MIN_REPORTS

    return {
        "recent_reports": recent_reports,
        "recent_reports_prev": prev_reports,
        "historical_incidence": historical_incidence,
        "spread_velocity": round(float(spread_velocity), 3),
        "rapid_spread": bool(rapid_spread),
    }


def aggregate_regions(
    df: pd.DataFrame,
    region_col: str,
    as_of_date=None,
) -> pd.DataFrame:
    """
    Roll individual reports up to one row per region (village / block /
    district), with the counts risk_scoring.py needs: short/long window
    volume, mortality, high-concern ratio, growth rate and historical
    incidence.
    """
    d = _to_datetime(df)
    if as_of_date is None:
        as_of_date = d["date"].max()
    as_of = pd.to_datetime(as_of_date)

    short_start = as_of - pd.Timedelta(days=config.ROLLING_WINDOW_SHORT_DAYS - 1)
    long_start = as_of - pd.Timedelta(days=config.ROLLING_WINDOW_LONG_DAYS - 1)
    prev_start = short_start - pd.Timedelta(days=config.ROLLING_WINDOW_SHORT_DAYS)
    prev_end = short_start - pd.Timedelta(days=1)

    rows = []
    for region_value, g in d.groupby(region_col):
        g_short = g[(g["date"] >= short_start) & (g["date"] <= as_of)]
        g_long = g[(g["date"] >= long_start) & (g["date"] <= as_of)]
        g_prev = g[(g["date"] >= prev_start) & (g["date"] <= prev_end)]
        g_hist = g[g["date"] < as_of]

        report_count_7d = len(g_short)
        report_count_30d = len(g_long)
        report_count_prev_7d = len(g_prev)
        affected_animals = int(g_short["number_affected"].sum())
        deaths = int(g_short["number_deaths"].sum())
        mortality_rate = (deaths / affected_animals) if affected_animals else 0.0
        high_concern_count = int((g_short["risk_level"] == "HIGH").sum())
        high_concern_ratio = (high_concern_count / report_count_7d) if report_count_7d else 0.0
        report_growth_rate = (report_count_7d / report_count_prev_7d) if report_count_prev_7d else float(report_count_7d)
        historical_incidence = int((g_hist["risk_level"] == "HIGH").sum())

        rows.append({
            region_col: region_value,
            "report_count_7d": report_count_7d,
            "report_count_30d": report_count_30d,
            "affected_animals": affected_animals,
            "deaths": deaths,
            "mortality_rate": round(mortality_rate, 4),
            "high_concern_count": high_concern_count,
            "high_concern_ratio": round(high_concern_ratio, 4),
            "report_growth_rate": round(report_growth_rate, 4),
            "historical_incidence": historical_incidence,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(columns=[
            region_col, "report_count_7d", "report_count_30d", "affected_animals",
            "deaths", "mortality_rate", "high_concern_count", "high_concern_ratio",
            "report_growth_rate", "historical_incidence",
        ])
    return result
