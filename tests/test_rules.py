import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import rules  # noqa: E402


def _report(**overrides):
    base = {
        "number_affected": 10,
        "number_deaths": 0,
        "symptom_neurological_symptoms": False,
        "symptom_sudden_death": False,
        "symptom_respiratory_distress": False,
    }
    base.update(overrides)
    return base


def test_no_rule_triggered_for_mild_report():
    result = rules.evaluate_rules(_report(number_affected=2, number_deaths=0))
    assert result.triggered is False
    assert result.rule_name is None


def test_multiple_deaths_rule():
    report = _report(number_affected=8, number_deaths=3)
    result = rules.evaluate_rules(report)
    assert result.triggered is True
    assert result.rule_name == "multiple_deaths"


def test_multiple_deaths_rule_does_not_fire_below_affected_threshold():
    # 3 deaths but fewer than the minimum affected-animal threshold
    report = _report(number_affected=4, number_deaths=3)
    result = rules.evaluate_rules(report)
    assert result.rule_name != "multiple_deaths"


def test_high_mortality_rate_rule():
    # Mortality rate >=30% but fewer than 3 deaths, so the multiple_deaths
    # rule doesn't fire first - isolates the mortality-rate rule specifically.
    report = _report(number_affected=6, number_deaths=2)  # 33% mortality
    result = rules.evaluate_rules(report)
    assert result.triggered is True
    assert result.rule_name == "high_mortality_rate"


def test_neurological_symptom_rule():
    report = _report(number_affected=5, number_deaths=0, symptom_neurological_symptoms=True)
    result = rules.evaluate_rules(report)
    assert result.triggered is True
    assert result.rule_name == "critical_symptoms"


def test_sudden_death_symptom_rule():
    report = _report(number_affected=5, number_deaths=0, symptom_sudden_death=True)
    result = rules.evaluate_rules(report)
    assert result.triggered is True
    assert result.rule_name == "critical_symptoms"


def test_respiratory_cluster_rule():
    report = _report(number_affected=4, number_deaths=0, symptom_respiratory_distress=True)
    result = rules.evaluate_rules(report)
    assert result.triggered is True
    assert result.rule_name == "respiratory_cluster"


def test_respiratory_rule_does_not_fire_for_single_animal():
    report = _report(number_affected=1, number_deaths=0, symptom_respiratory_distress=True)
    result = rules.evaluate_rules(report)
    assert result.rule_name != "respiratory_cluster"


def test_rapid_spread_flag_triggers_rule():
    report = _report(number_affected=2, number_deaths=0)
    result = rules.evaluate_rules(report, rapid_spread=True)
    assert result.triggered is True
    assert result.rule_name == "rapid_spread"
