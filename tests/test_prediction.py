import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src import config, predictor  # noqa: E402


@pytest.fixture(scope="module")
def history_df():
    if not os.path.exists(config.REPORTS_CSV):
        pytest.skip("Synthetic dataset not generated - run `python src/data_generator.py` first.")
    return pd.read_csv(config.REPORTS_CSV)


@pytest.fixture(scope="module")
def model_ready():
    if not (os.path.exists(config.MODEL_PATH) and os.path.exists(config.PREPROCESSOR_PATH)):
        pytest.skip("Model not trained - run `python src/train_model.py` first.")
    return True


def _base_report(**overrides):
    report = {
        "date": "2026-09-03", "district": "Nashik", "block": "Nashik Block-1",
        "village": "Nashik Block-1 Village-1", "species": "cattle",
        "number_affected": 2, "number_deaths": 0, "duration_days": 2,
        "temperature": 38.5, "notes": "",
    }
    for s in config.SYMPTOM_COLUMNS:
        report[f"symptom_{s}"] = False
    report.update(overrides)
    return report


def test_prediction_pipeline_runs_without_error(history_df, model_ready):
    report = _base_report()
    result = predictor.predict_report(report, history_df)
    assert result["valid"] is True
    assert result["final_risk"] in config.RISK_LEVELS
    assert 0.0 <= result["ml_confidence"] <= 1.0
    assert isinstance(result["explanation"], list) and len(result["explanation"]) > 0


def test_red_flag_report_is_forced_high(history_df, model_ready):
    report = _base_report(number_affected=10, number_deaths=6)  # 60% mortality
    result = predictor.predict_report(report, history_df)
    assert result["valid"] is True
    assert result["rule_triggered"] is True
    assert result["final_risk"] == "HIGH"


def test_neurological_symptom_forces_high(history_df, model_ready):
    report = _base_report(symptom_neurological_symptoms=True)
    result = predictor.predict_report(report, history_df)
    assert result["final_risk"] == "HIGH"
    assert result["rule_name"] == "critical_symptoms"


def test_mild_report_is_not_forced_high(history_df, model_ready):
    report = _base_report(number_affected=1, number_deaths=0, symptom_weakness=True)
    result = predictor.predict_report(report, history_df)
    assert result["rule_triggered"] is False


def test_invalid_negative_affected_count(history_df, model_ready):
    report = _base_report(number_affected=-5)
    result = predictor.predict_report(report, history_df)
    assert result["valid"] is False
    assert any("positive integer" in e for e in result["errors"])


def test_invalid_deaths_exceed_affected(history_df, model_ready):
    report = _base_report(number_affected=2, number_deaths=10)
    result = predictor.predict_report(report, history_df)
    assert result["valid"] is False
    assert any("cannot exceed" in e for e in result["errors"])


def test_invalid_species(history_df, model_ready):
    report = _base_report(species="dragon")
    result = predictor.predict_report(report, history_df)
    assert result["valid"] is False
    assert any("Species must be one of" in e for e in result["errors"])


def test_missing_location_fields(history_df, model_ready):
    report = _base_report(district="", block="", village="")
    result = predictor.predict_report(report, history_df)
    assert result["valid"] is False
    assert any("District, block and village" in e for e in result["errors"])
