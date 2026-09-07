# Livestock Health Early Warning System (MVP)

A prototype for **SIH Problem Statement 26128** - *Efficient systems for early
detection, prevention and management of livestock diseases and animal health
issues* (Government of Maharashtra).

> **All health reports and disease observations in this MVP are
> synthetic/simulated.** Real village-level livestock disease data is not
> publicly available at the granularity this problem needs, so the dataset
> was generated to *resemble* plausible reports (realistic correlations
> between mortality, symptoms and risk, with injected outbreak clusters over
> time) purely for model-development and demonstration purposes. The
> dashboard displays a permanent "SYNTHETIC DEMO DATA" banner so this is
> never mistaken for real surveillance data. District names are real
> Maharashtra districts used only as geographic anchors; blocks, villages and
> every report are synthetic.

This is a **triage / early-warning system, not a diagnostic tool** - it
flags which reports deserve faster veterinary attention. It never claims to
identify a specific disease.

---

## What it does

1. **Individual report triage** - an ML model (with a rule-based safety net)
   classifies each incoming report as LOW / MEDIUM / HIGH concern.
2. **Village/block/district risk aggregation** - a transparent, weighted
   formula rolls individual reports up into a regional risk score.
3. **Geographical hotspot map** - risk scores plotted on an interactive map,
   color-coded by category.
4. **Official dashboard** - KPIs, recent reports, regional ranking, trends,
   and a live report-submission form, all in Streamlit.

## Project structure

```
livestock-ai-mvp/
├── app.py                    # Streamlit dashboard (entry point)
├── requirements.txt
├── README.md
├── data/
│   ├── livestock_reports.csv # synthetic reports (generated)
│   └── regions.csv           # district/block/village + coordinates (generated)
├── models/
│   ├── risk_model.pkl        # trained classifier (generated)
│   └── preprocessor.pkl      # fitted sklearn preprocessor (generated)
├── src/
│   ├── config.py             # every threshold/weight/path, in one place
│   ├── data_generator.py     # synthetic data generation
│   ├── preprocessing.py      # feature engineering + sklearn pipeline
│   ├── train_model.py        # trains + evaluates + saves the model
│   ├── predictor.py          # ML + rules end-to-end prediction pipeline
│   ├── rules.py              # rule-based safety fallback
│   ├── risk_scoring.py       # regional 0-100 risk formula
│   ├── aggregation.py        # rolling-window regional rollups
│   └── map_utils.py          # Folium hotspot map builder
└── tests/
    ├── test_rules.py
    ├── test_risk_scoring.py
    └── test_prediction.py
```

## Setup and run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the synthetic dataset (data/regions.csv, data/livestock_reports.csv)
python src/data_generator.py

# 3. Train the triage model (models/risk_model.pkl, models/preprocessor.pkl)
python src/train_model.py

# 4. Run the test suite
pytest

# 5. Start the dashboard
streamlit run app.py
```

Steps 2 and 3 only need to be re-run if you want to regenerate the dataset or
retrain the model; the dashboard loads the saved artifacts on startup.

## The ML pipeline, briefly

- **Model**: `RandomForestClassifier` (`class_weight="balanced"`), not a deep
  learning model - the dataset size and feature count don't call for one, and
  a simpler model is easier to explain to judges and to safely override.
- **Features**: species, affected/deaths/mortality rate, duration, optional
  temperature, per-symptom flags, an aggregate symptom-severity score, and
  village-level context (recent report count, historical HIGH-concern count,
  spread velocity) computed only from reports *before* the one being scored,
  so there's no label leakage.
- **Target**: `risk_level` (LOW/MEDIUM/HIGH) — assigned in the synthetic
  dataset by a documented formula with injected noise, so it's genuinely
  learnable rather than trivially deterministic.
- **Evaluation** (on the generated dataset, 20% held-out test split):
  ~92% accuracy, and — the metric that actually matters for this problem —
  **~81% recall on HIGH-concern reports**. Full confusion matrix, precision/
  recall/F1 per class, and top feature importances print when you run
  `train_model.py`.
- **Safety net**: `rules.py` runs independently of the model and can force a
  report to HIGH regardless of the ML prediction (multiple deaths, mortality
  rate ≥30%, neurological symptoms or sudden death, a respiratory cluster, or
  a rapid spread of reports from the same village). The final result always
  shows both the ML prediction and whether a rule overrode it, with a
  plain-language reason.

## The regional risk formula

```
Regional Risk Score (0-100) =
      25% report volume         (reports in the last 7 days)
    + 20% recent growth         (7-day count vs. the prior 7-day count)
    + 20% mortality             (mortality rate in the last 7 days)
    + 20% high-concern ratio    (share of last-7-day reports rated HIGH)
    + 15% historical incidence  (past HIGH-concern reports for this region)
```

Each component is normalized to 0-100 against a documented cap
(`config.NORMALIZATION_CAPS`) before the weights are applied, so no single
saturating input dominates the score. Categories: 0-29 LOW, 30-59 MEDIUM,
60-79 HIGH, 80-100 CRITICAL. Weights, caps and the rolling windows (7-day and
30-day) are all defined once in `src/config.py` — nothing is hard-coded
elsewhere.

## Demonstrating this to judges

1. Open the dashboard - the KPI row and hotspot map already show a handful of
   simulated outbreak clusters, so there's something visually meaningful
   immediately (no need to submit anything first).
2. Open the **Risk Overview** tab to show the top-risk villages with a
   plain-language breakdown of *why* each one is flagged.
3. Go to **Submit New Report**, enter a report with an obvious red flag
   (e.g. several deaths out of a small affected count), and show the result
   card: the ML score, whether the safety rule overrode it, and the
   recommended action.
4. Go to the **Hotspot Map** tab and click **Simulate New Outbreak** on a
   village of your choice. Each generated report is triaged live through the
   real pipeline, so you can walk judges through the story as it happens:
   individual reports come in → high-concern reports increase → the
   village's risk score climbs → the hotspot becomes visible on the map -
   the whole early-warning loop, live.
5. Use the **Regional Risk Ranking** and **Trends** tabs to show this scales
   from village to block to district, and that the system tracks change over
   time rather than just a single snapshot.

Two things worth being upfront about if asked: this MVP triages/aggregates
rather than diagnoses a specific disease, and it runs on synthetic data
because real village-level surveillance data isn't available for a
build like this - both are reasonable, common scoping choices for a
hackathon prototype, and the architecture (feature list, rule set, risk
formula) is designed to carry over directly once real data is available.
