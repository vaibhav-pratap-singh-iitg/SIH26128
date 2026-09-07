import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src import risk_scoring  # noqa: E402


def _reports(rows):
    return pd.DataFrame(rows)


def test_village_with_no_reports_scores_zero():
    df = _reports([
        {"village": "A", "block": "B1", "district": "D1", "date": "2026-01-01",
         "number_affected": 1, "number_deaths": 0, "risk_level": "LOW"},
    ])
    ranked = risk_scoring.calculate_village_risk(df, as_of_date="2026-03-01")
    row = ranked[ranked["village"] == "A"].iloc[0]
    assert row["risk_score"] == 0.0
    assert row["risk_category"] == "LOW"


def test_high_mortality_increases_score():
    low_mortality = _reports([
        {"village": "A", "block": "B1", "district": "D1", "date": "2026-01-05",
         "number_affected": 10, "number_deaths": 0, "risk_level": "LOW"},
    ])
    high_mortality = _reports([
        {"village": "B", "block": "B1", "district": "D1", "date": "2026-01-05",
         "number_affected": 10, "number_deaths": 5, "risk_level": "HIGH"},
    ])
    as_of = "2026-01-05"
    score_low = risk_scoring.calculate_village_risk(low_mortality, as_of_date=as_of).iloc[0]["risk_score"]
    score_high = risk_scoring.calculate_village_risk(high_mortality, as_of_date=as_of).iloc[0]["risk_score"]
    assert score_high > score_low


def test_report_growth_increases_score():
    # Village with a burst of recent reports vs. a steady trickle
    steady = [{"village": "A", "block": "B1", "district": "D1", "date": d,
               "number_affected": 1, "number_deaths": 0, "risk_level": "LOW"}
              for d in ["2026-01-01", "2026-01-08", "2026-01-15"]]
    burst = [{"village": "B", "block": "B1", "district": "D1", "date": "2026-01-14",
              "number_affected": 1, "number_deaths": 0, "risk_level": "LOW"}
             for _ in range(6)]
    as_of = "2026-01-15"
    steady_score = risk_scoring.calculate_village_risk(_reports(steady), as_of_date=as_of)
    steady_score = steady_score[steady_score["village"] == "A"].iloc[0]["risk_score"]
    burst_score = risk_scoring.calculate_village_risk(_reports(burst), as_of_date=as_of)
    burst_score = burst_score[burst_score["village"] == "B"].iloc[0]["risk_score"]
    assert burst_score > steady_score


def test_district_and_block_aggregation_run_without_error():
    df = _reports([
        {"village": "A", "block": "B1", "district": "D1", "date": "2026-01-01",
         "number_affected": 2, "number_deaths": 1, "risk_level": "HIGH"},
        {"village": "C", "block": "B2", "district": "D1", "date": "2026-01-02",
         "number_affected": 3, "number_deaths": 0, "risk_level": "LOW"},
    ])
    block_risk = risk_scoring.calculate_block_risk(df, as_of_date="2026-01-05")
    district_risk = risk_scoring.calculate_district_risk(df, as_of_date="2026-01-05")
    assert set(block_risk["block"]) == {"B1", "B2"}
    assert set(district_risk["district"]) == {"D1"}


def test_risk_category_thresholds():
    from src import config
    assert config.risk_score_to_category(0) == "LOW"
    assert config.risk_score_to_category(29.9) == "LOW"
    assert config.risk_score_to_category(30) == "MEDIUM"
    assert config.risk_score_to_category(59.9) == "MEDIUM"
    assert config.risk_score_to_category(60) == "HIGH"
    assert config.risk_score_to_category(79.9) == "HIGH"
    assert config.risk_score_to_category(80) == "CRITICAL"
    assert config.risk_score_to_category(100) == "CRITICAL"
