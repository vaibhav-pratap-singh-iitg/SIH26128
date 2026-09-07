"""
Synthetic data generator for the Livestock Health Early Warning System MVP.

*** ALL DATA PRODUCED BY THIS MODULE IS SYNTHETIC / SIMULATED. ***
It is designed to *resemble* plausible livestock health reports so the
triage model and risk-scoring pipeline have something realistic to learn
from and demonstrate on. It must never be presented as real surveillance
data. District names are real Maharashtra districts (used only as
geographic anchors); block and village names, and every report, are
synthetic.

Run directly to (re)generate data/regions.csv and data/livestock_reports.csv:

    python src/data_generator.py
"""

from __future__ import annotations

import os
import random
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

# Reference "today" for the synthetic timeline - matches the hackathon's
# working date so the demo data feels current.
END_DATE = pd.Timestamp("2026-09-03")
HISTORY_DAYS = 120
START_DATE = END_DATE - timedelta(days=HISTORY_DAYS)

# Real Maharashtra districts used only as geographic anchors (approx centroid
# lat/lon). Blocks and villages beneath them are synthetic labels.
DISTRICT_CENTROIDS = {
    "Nashik": (20.0059, 73.7910),
    "Pune": (18.5204, 73.8567),
    "Nagpur": (21.1458, 79.0882),
    "Kolhapur": (16.7050, 74.2433),
    "Sambhajinagar": (19.8762, 75.3433),
    "Amravati": (20.9320, 77.7523),
}

BLOCKS_PER_DISTRICT = 3
VILLAGES_PER_BLOCK = 5

# Number of villages that will experience a simulated outbreak cluster.
N_OUTBREAK_CLUSTERS = 10
OUTBREAK_MIN_LEN_DAYS = 8
OUTBREAK_MAX_LEN_DAYS = 20

N_BASELINE_REPORTS = 5200  # additional non-outbreak reports spread across the period


def _rng() -> np.random.Generator:
    return np.random.default_rng(config.RANDOM_SEED)


def build_regions() -> pd.DataFrame:
    """Build the District -> Block -> Village hierarchy with jittered coordinates."""
    rng = _rng()
    rows = []
    for district, (clat, clon) in DISTRICT_CENTROIDS.items():
        for b in range(1, BLOCKS_PER_DISTRICT + 1):
            block = f"{district} Block-{b}"
            block_lat = clat + rng.uniform(-0.15, 0.15)
            block_lon = clon + rng.uniform(-0.15, 0.15)
            for v in range(1, VILLAGES_PER_BLOCK + 1):
                village = f"{block} Village-{v}"
                lat = block_lat + rng.uniform(-0.06, 0.06)
                lon = block_lon + rng.uniform(-0.06, 0.06)
                rows.append({
                    "district": district,
                    "block": block,
                    "village": village,
                    "latitude": round(float(lat), 5),
                    "longitude": round(float(lon), 5),
                })
    return pd.DataFrame(rows)


def _pick_symptoms(rng: np.random.Generator, severity_level: str) -> dict:
    """Sample a plausible symptom set. severity_level in {'low','medium','high'}."""
    base_probs = {
        "fever": 0.35, "coughing": 0.25, "nasal_discharge": 0.20, "weakness": 0.25,
        "diarrhea": 0.15, "loss_of_appetite": 0.20, "reduced_milk_production": 0.15,
        "skin_lesions": 0.08, "swelling": 0.08, "respiratory_distress": 0.06,
        "neurological_symptoms": 0.02, "sudden_death": 0.02,
    }
    boost = {"low": 1.0, "medium": 1.8, "high": 3.2}[severity_level]
    symptoms = {}
    for s, p in base_probs.items():
        prob = min(0.95, p * boost)
        symptoms[s] = bool(rng.random() < prob)
    # Guarantee at least one symptom is present
    if not any(symptoms.values()):
        symptoms[rng.choice(list(base_probs.keys()))] = True
    return symptoms


def _simulate_report(rng: np.random.Generator, village_row: pd.Series, date: pd.Timestamp,
                      severity_level: str, report_id: int) -> dict:
    species = rng.choice(config.SPECIES, p=[0.32, 0.18, 0.24, 0.16, 0.10])

    if severity_level == "high":
        number_affected = int(rng.integers(4, 25))
        mortality_rate_target = rng.uniform(0.15, 0.55)
    elif severity_level == "medium":
        number_affected = int(rng.integers(2, 10))
        mortality_rate_target = rng.uniform(0.03, 0.20)
    else:
        number_affected = int(rng.integers(1, 5))
        mortality_rate_target = rng.uniform(0.0, 0.06)

    number_deaths = int(round(number_affected * mortality_rate_target))
    number_deaths = min(number_deaths, number_affected)

    symptoms = _pick_symptoms(rng, severity_level)
    duration_days = int(rng.integers(1, 4)) if severity_level == "high" else int(rng.integers(1, 10))

    has_fever = symptoms["fever"]
    if has_fever:
        temperature = round(float(rng.uniform(39.3, 41.5)), 1)
    else:
        temperature = round(float(rng.uniform(37.6, 39.2)), 1)
    if rng.random() < 0.30:  # optional field - sometimes not recorded
        temperature = np.nan

    notes_pool = [
        "", "", "", "Reported by local field worker.", "Farmer requested urgent visit.",
        "Similar cases seen in neighbouring herd.", "Animals isolated as precaution.",
    ]
    notes = str(rng.choice(notes_pool))

    row = {
        "report_id": report_id,
        "date": date.strftime("%Y-%m-%d"),
        "district": village_row["district"],
        "block": village_row["block"],
        "village": village_row["village"],
        "species": species,
        "number_affected": number_affected,
        "number_deaths": number_deaths,
        "duration_days": duration_days,
        "temperature": temperature,
        "notes": notes,
    }
    for s in config.SYMPTOM_COLUMNS:
        row[f"symptom_{s}"] = symptoms[s]
    return row


def build_reports(regions: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic reports with injected outbreak clusters over time."""
    rng = _rng()
    py_rng = random.Random(config.RANDOM_SEED)

    reports = []
    report_id = 1

    # --- 1. Outbreak clusters: a handful of villages get a burst of
    #        medium/high severity reports over a contiguous date window. ---
    outbreak_villages = regions.sample(n=N_OUTBREAK_CLUSTERS, random_state=config.RANDOM_SEED)
    for _, village_row in outbreak_villages.iterrows():
        cluster_len = py_rng.randint(OUTBREAK_MIN_LEN_DAYS, OUTBREAK_MAX_LEN_DAYS)
        latest_start = HISTORY_DAYS - cluster_len - 1
        start_offset = py_rng.randint(0, max(1, latest_start))
        cluster_start = START_DATE + timedelta(days=start_offset)

        n_cluster_reports = py_rng.randint(12, 30)
        for _ in range(n_cluster_reports):
            day_offset = py_rng.randint(0, cluster_len)
            date = cluster_start + timedelta(days=day_offset)
            # Severity ramps up then tapers across the cluster window.
            progress = day_offset / max(1, cluster_len)
            if progress < 0.6:
                severity_level = py_rng.choices(["medium", "high"], weights=[0.45, 0.55])[0]
            else:
                severity_level = py_rng.choices(["medium", "high"], weights=[0.65, 0.35])[0]
            reports.append(_simulate_report(rng, village_row, date, severity_level, report_id))
            report_id += 1

    # --- 2. Baseline reports: scattered across all villages and dates,
    #        overwhelmingly low/medium severity - normal background noise. ---
    for _ in range(N_BASELINE_REPORTS):
        village_row = regions.iloc[py_rng.randrange(len(regions))]
        day_offset = py_rng.randint(0, HISTORY_DAYS)
        date = START_DATE + timedelta(days=day_offset)
        severity_level = py_rng.choices(["low", "medium", "high"], weights=[0.78, 0.18, 0.04])[0]
        reports.append(_simulate_report(rng, village_row, date, severity_level, report_id))
        report_id += 1

    df = pd.DataFrame(reports)
    df["mortality_rate"] = (df["number_deaths"] / df["number_affected"]).round(3)
    df = df.sort_values("date").reset_index(drop=True)
    df["report_id"] = range(1, len(df) + 1)

    # --- Label generation: a hidden "ground truth" risk formula with noise,
    #     used purely to create a learnable supervised-learning target. ---
    symptom_cols = [f"symptom_{s}" for s in config.SYMPTOM_COLUMNS]
    severity_weight = np.array([config.SYMPTOM_SEVERITY[s] for s in config.SYMPTOM_COLUMNS])
    symptom_severity = df[symptom_cols].values.astype(float) @ severity_weight
    symptom_severity_norm = symptom_severity / severity_weight.sum()

    affected_norm = np.clip(df["number_affected"] / 20.0, 0, 1)
    mortality_norm = np.clip(df["mortality_rate"].fillna(0) / 0.5, 0, 1)
    critical_flag = df["symptom_neurological_symptoms"] | df["symptom_sudden_death"]

    noise = rng.normal(0, 0.06, size=len(df))
    risk_raw = (
        0.35 * mortality_norm
        + 0.25 * symptom_severity_norm
        + 0.20 * affected_norm
        + 0.20 * critical_flag.astype(float)
        + noise
    )
    df["_risk_raw"] = risk_raw
    df["risk_level"] = pd.cut(
        risk_raw, bins=[-np.inf, 0.28, 0.48, np.inf], labels=["LOW", "MEDIUM", "HIGH"]
    ).astype(str)
    df = df.drop(columns=["_risk_raw"])

    return df


def generate_and_save() -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    regions = build_regions()
    reports = build_reports(regions)

    regions.to_csv(config.REGIONS_CSV, index=False)
    reports.to_csv(config.REPORTS_CSV, index=False)

    print(f"Generated {len(regions)} regions -> {config.REGIONS_CSV}")
    print(f"Generated {len(reports)} reports -> {config.REPORTS_CSV}")
    print("Risk level distribution:")
    print(reports["risk_level"].value_counts())
    print("\nNOTE: this dataset is entirely synthetic/simulated and is for "
          "demonstration and model-development purposes only.")


if __name__ == "__main__":
    generate_and_save()
