"""
Livestock Health Early Warning System - Streamlit dashboard.

Run with:
    streamlit run app.py

*** All data shown is SYNTHETIC / SIMULATED - see README.md. ***
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from src import aggregation, config, map_utils, predictor, risk_scoring

st.set_page_config(page_title="Livestock Health Early Warning System", page_icon="\U0001F404", layout="wide")

CATEGORY_EMOJI = {"LOW": "\U0001F7E2", "MEDIUM": "\U0001F7E1", "HIGH": "\U0001F7E0", "CRITICAL": "\U0001F534"}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data
def load_regions() -> pd.DataFrame:
    return pd.read_csv(config.REGIONS_CSV)


@st.cache_data
def load_base_reports() -> pd.DataFrame:
    df = pd.read_csv(config.REPORTS_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df


def init_state():
    if "reports_df" not in st.session_state:
        st.session_state.reports_df = load_base_reports().copy()
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


regions_df = load_regions()
init_state()

try:
    predictor._load_artifacts()
    MODEL_READY = True
except predictor.ModelNotTrainedError:
    MODEL_READY = False


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("\U0001F404 Livestock Health Early Warning System")
st.caption("AI-assisted surveillance and regional risk monitoring")
st.warning(
    "**SYNTHETIC DEMO DATA** - every report and outbreak in this prototype is simulated for "
    "demonstration and model-development purposes. Nothing here reflects a real disease event.",
    icon="\u26A0\uFE0F",
)

if not MODEL_READY:
    st.error(
        "No trained model found. Run `python src/data_generator.py` then "
        "`python src/train_model.py` before using this dashboard."
    )
    st.stop()

reports_df = st.session_state.reports_df


# --------------------------------------------------------------------------
# Outbreak simulation helper (defined early - used by the Hotspot Map tab)
# --------------------------------------------------------------------------
def simulate_outbreak(village: str, n_reports: int) -> None:
    """
    Generate a burst of new reports for `village`, each triaged sequentially
    through the real predict_report pipeline so recent-report context
    (and therefore risk) escalates realistically as the cluster grows.
    """
    rng = np.random.default_rng()
    village_row = regions_df[regions_df["village"] == village].iloc[0]
    base_date = pd.Timestamp(st.session_state.reports_df["date"].max())

    severe_symptoms = ["fever", "respiratory_distress", "weakness", "loss_of_appetite"]
    critical_symptoms = ["neurological_symptoms", "sudden_death"]

    for i in range(n_reports):
        progress = i / max(1, n_reports - 1)
        affected = int(rng.integers(2, 6) + progress * 12)
        mortality_target = 0.05 + progress * 0.35
        deaths = min(affected, int(round(affected * mortality_target)))

        symptoms = {s: False for s in config.SYMPTOM_COLUMNS}
        for s in severe_symptoms:
            if rng.random() < 0.5 + 0.3 * progress:
                symptoms[s] = True
        for s in critical_symptoms:
            if rng.random() < 0.15 * progress:
                symptoms[s] = True
        if not any(symptoms.values()):
            symptoms["fever"] = True

        report = {
            "date": (base_date + pd.Timedelta(days=int(i // 3))).strftime("%Y-%m-%d"),
            "district": village_row["district"], "block": village_row["block"], "village": village,
            "species": str(rng.choice(config.SPECIES)),
            "number_affected": affected, "number_deaths": deaths,
            "duration_days": int(rng.integers(1, 4)),
            "temperature": round(float(rng.uniform(39.0, 41.5)), 1),
            "notes": "Simulated outbreak report (synthetic demo).",
        }
        for s in config.SYMPTOM_COLUMNS:
            report[f"symptom_{s}"] = symptoms[s]

        result = predictor.predict_report(report, st.session_state.reports_df)
        if not result["valid"]:
            continue
        new_row = dict(report)
        new_row["mortality_rate"] = round(deaths / affected, 4) if affected else 0.0
        new_row["risk_level"] = result["final_risk"]
        new_row["report_id"] = int(st.session_state.reports_df["report_id"].max()) + 1
        new_row["date"] = pd.Timestamp(new_row["date"])
        st.session_state.reports_df = pd.concat(
            [st.session_state.reports_df, pd.DataFrame([new_row])], ignore_index=True
        )
        st.session_state.reports_df["date"] = pd.to_datetime(st.session_state.reports_df["date"])


# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------
st.sidebar.header("Filters")

districts = sorted(reports_df["district"].unique())
sel_districts = st.sidebar.multiselect("District", districts, default=districts)

block_options = sorted(reports_df[reports_df["district"].isin(sel_districts)]["block"].unique())
sel_blocks = st.sidebar.multiselect("Block", block_options, default=block_options)

sel_species = st.sidebar.multiselect("Species", config.SPECIES, default=config.SPECIES)

min_date, max_date = reports_df["date"].min().date(), reports_df["date"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

base_filtered = reports_df[
    reports_df["district"].isin(sel_districts)
    & reports_df["block"].isin(sel_blocks)
    & reports_df["species"].isin(sel_species)
    & (reports_df["date"].dt.date >= start_date)
    & (reports_df["date"].dt.date <= end_date)
].copy()

as_of = base_filtered["date"].max() if not base_filtered.empty else reports_df["date"].max()

st.sidebar.caption(
    f"Showing {len(base_filtered):,} of {len(reports_df):,} reports. "
    f"Risk scores computed as of {pd.Timestamp(as_of).date()}."
)

# --------------------------------------------------------------------------
# KPI cards
# --------------------------------------------------------------------------
village_risk = risk_scoring.calculate_village_risk(base_filtered, as_of_date=as_of) if not base_filtered.empty else pd.DataFrame()
village_risk = village_risk.merge(regions_df, on="village", how="left", suffixes=("", "_region"))
if "district_region" in village_risk.columns:
    village_risk["district"] = village_risk["district"].fillna(village_risk["district_region"])
    village_risk["block"] = village_risk["block"].fillna(village_risk["block_region"])

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Reports", f"{len(base_filtered):,}")
k2.metric("Affected Animals", f"{int(base_filtered['number_affected'].sum()):,}")
k3.metric("Deaths", f"{int(base_filtered['number_deaths'].sum()):,}")
k4.metric("High-Concern Reports", f"{int((base_filtered['risk_level'] == 'HIGH').sum()):,}")
n_high_risk_regions = int((village_risk["risk_category"].isin(["HIGH", "CRITICAL"])).sum()) if not village_risk.empty else 0
k5.metric("High-Risk Regions", f"{n_high_risk_regions:,}")

st.divider()

tabs = st.tabs([
    "Risk Overview", "Hotspot Map", "Recent Reports",
    "Regional Risk Ranking", "Trends", "Submit New Report",
])

# --------------------------------------------------------------------------
# Tab 1: Risk Overview
# --------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Top regions by risk score")
    if village_risk.empty:
        st.info("No reports match the current filters.")
    else:
        top5 = village_risk.head(5)
        for _, row in top5.iterrows():
            emoji = CATEGORY_EMOJI.get(row["risk_category"], "\U0001F7E2")
            with st.expander(f"{emoji} {row['village']}  \u2014  score {row['risk_score']:.0f}/100 ({row['risk_category']})"):
                st.write(f"**{row.get('block', '')}, {row.get('district', '')}**")
                for reason in risk_scoring.explain_region_risk(row):
                    st.markdown(f"- {reason}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Reports (7d)", int(row["report_count_7d"]))
                c2.metric("Affected", int(row["affected_animals"]))
                c3.metric("Deaths", int(row["deaths"]))
                c4.metric("HIGH reports", int(row["high_concern_count"]))

# --------------------------------------------------------------------------
# Tab 2: Hotspot Map
# --------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Geographical hotspot map")
    if village_risk.empty:
        st.info("No reports match the current filters.")
    else:
        fmap = map_utils.build_hotspot_map(village_risk)
        st_folium(fmap, width=None, height=520, returned_objects=[])

    st.divider()
    st.subheader("Demo: simulate a new outbreak")
    st.caption(
        "Generates a burst of new reports in one village, each triaged live through the same "
        "model + rule pipeline used above, so you can watch report volume, high-concern counts "
        "and the region's risk score climb in real time."
    )
    sim_col1, sim_col2, sim_col3 = st.columns([2, 1, 1])
    all_villages = sorted(regions_df["village"].unique())
    sim_village = sim_col1.selectbox("Village", all_villages, key="sim_village")
    sim_count = sim_col2.number_input("Number of reports", min_value=3, max_value=25, value=10, step=1)
    if sim_col3.button("\U0001F6A8 Simulate New Outbreak", type="primary"):
        simulate_outbreak(sim_village, int(sim_count))
        st.rerun()

# --------------------------------------------------------------------------
# Tab 3: Recent Reports
# --------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Recent reports")
    risk_filter = st.multiselect(
        "Report risk level (this table only)", config.RISK_LEVELS, default=config.RISK_LEVELS,
    )
    table = base_filtered[base_filtered["risk_level"].isin(risk_filter)].sort_values("date", ascending=False)
    display_cols = [
        "date", "district", "block", "village", "species", "number_affected",
        "number_deaths", "mortality_rate", "risk_level", "notes",
    ]
    st.dataframe(table[display_cols].head(200), use_container_width=True, hide_index=True)
    st.caption(f"Showing up to 200 of {len(table):,} matching reports, most recent first.")

# --------------------------------------------------------------------------
# Tab 4: Regional Risk Ranking
# --------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Regional risk ranking")
    granularity = st.radio("Granularity", ["Village", "Block", "District"], horizontal=True)
    calc_fn = {
        "Village": risk_scoring.calculate_village_risk,
        "Block": risk_scoring.calculate_block_risk,
        "District": risk_scoring.calculate_district_risk,
    }[granularity]
    ranked = calc_fn(base_filtered, as_of_date=as_of) if not base_filtered.empty else pd.DataFrame()
    if ranked.empty:
        st.info("No reports match the current filters.")
    else:
        ranked = ranked.copy()
        ranked.insert(0, "", ranked["risk_category"].map(CATEGORY_EMOJI))
        show_cols = [
            "", granularity.lower(), "risk_score", "risk_category", "report_count_7d",
            "report_count_30d", "affected_animals", "deaths", "mortality_rate",
            "high_concern_count", "report_growth_rate", "historical_incidence",
        ]
        st.dataframe(ranked[show_cols], use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# Tab 5: Trends
# --------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Reporting trend")
    if base_filtered.empty:
        st.info("No reports match the current filters.")
    else:
        daily = base_filtered.groupby(base_filtered["date"].dt.date).agg(
            reports=("report_id", "count"),
            high_concern=("risk_level", lambda s: (s == "HIGH").sum()),
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["reports"], mode="lines", name="Total reports"))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["high_concern"], mode="lines", name="High-concern reports"))
        fig.update_layout(
            xaxis_title="Date", yaxis_title="Number of reports",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Tab 6: Submit New Report
# --------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Submit a new health report")
    st.caption(
        "This is a triage / early-warning tool, not a diagnostic system - it flags reports that "
        "deserve faster veterinary attention, and never claims to identify a specific disease."
    )

    c1, c2, c3 = st.columns(3)
    r_date = c1.date_input("Date", value=pd.Timestamp(as_of).date())
    r_district = c2.selectbox("District", sorted(regions_df["district"].unique()), key="new_district")
    r_block_opts = sorted(regions_df[regions_df["district"] == r_district]["block"].unique())
    r_block = c3.selectbox("Block", r_block_opts, key="new_block")
    r_village_opts = sorted(regions_df[(regions_df["district"] == r_district) & (regions_df["block"] == r_block)]["village"].unique())

    c4, c5, c6 = st.columns(3)
    r_village = c4.selectbox("Village", r_village_opts, key="new_village")
    r_species = c5.selectbox("Species", config.SPECIES, key="new_species")
    r_duration = c6.number_input("Duration of symptoms (days)", min_value=1, max_value=60, value=2)

    c7, c8, c9 = st.columns(3)
    r_affected = c7.number_input("Number of affected animals", min_value=1, max_value=500, value=1)
    r_deaths = c8.number_input("Number of deaths", min_value=0, max_value=500, value=0)
    r_temp = c9.number_input("Temperature (\u00b0C, optional - 0 = not recorded)", min_value=0.0, max_value=45.0, value=0.0, step=0.1)

    symptom_labels = {s: s.replace("_", " ").title() for s in config.SYMPTOM_COLUMNS}
    chosen = st.multiselect("Symptoms observed", list(symptom_labels.values()))
    chosen_keys = [k for k, v in symptom_labels.items() if v in chosen]

    r_notes = st.text_area("Additional notes (optional)", "")

    if st.button("Submit Report", type="primary"):
        report = {
            "date": pd.Timestamp(r_date).strftime("%Y-%m-%d"),
            "district": r_district, "block": r_block, "village": r_village,
            "species": r_species, "number_affected": int(r_affected), "number_deaths": int(r_deaths),
            "duration_days": int(r_duration),
            "temperature": float(r_temp) if r_temp > 0 else np.nan,
            "notes": r_notes,
        }
        for s in config.SYMPTOM_COLUMNS:
            report[f"symptom_{s}"] = s in chosen_keys

        result = predictor.predict_report(report, st.session_state.reports_df)
        st.session_state.last_result = result

        if result["valid"]:
            new_row = dict(report)
            new_row["mortality_rate"] = round(report["number_deaths"] / report["number_affected"], 4) if report["number_affected"] else 0.0
            new_row["risk_level"] = result["final_risk"]
            new_row["report_id"] = int(st.session_state.reports_df["report_id"].max()) + 1
            new_row["date"] = pd.Timestamp(new_row["date"])
            st.session_state.reports_df = pd.concat(
                [st.session_state.reports_df, pd.DataFrame([new_row])], ignore_index=True
            )
            st.session_state.reports_df["date"] = pd.to_datetime(st.session_state.reports_df["date"])

    result = st.session_state.last_result
    if result is not None:
        if not result["valid"]:
            st.error("Please fix the following before submitting:")
            for e in result["errors"]:
                st.markdown(f"- {e}")
        else:
            risk = result["final_risk"]
            box = {"LOW": st.success, "MEDIUM": st.warning, "HIGH": st.error}[risk]
            box(f"**FINAL RISK: {risk}**  |  ML score: {result['ml_confidence']:.0%}  |  "
                f"Safety rule: {'Triggered' if result['rule_triggered'] else 'Not triggered'}")
            if result["rule_triggered"]:
                st.markdown(f"**Rule reason:** {result['rule_explanation']}")
            st.markdown(f"**Recommended action:** {result['recommended_action']}")
            st.markdown("**Factors contributing to the model score:**")
            for line in result["explanation"]:
                st.markdown(f"- {line}")
            st.caption(
                "This report has been added to the live view for this session (filters, map and "
                "tables above will reflect it)."
            )
