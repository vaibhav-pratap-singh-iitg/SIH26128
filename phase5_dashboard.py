import json
import os
import pandas as pd
import streamlit as st

# =====================================================================
# DASHBOARD CONFIGURATION & DATA LOADING
# =====================================================================
st.set_page_config(
    page_title="Livestock Health Early Warning System",
    page_icon="🐄",
    layout="wide",
)


@st.cache_data
def load_surveillance_data():
    if not os.path.exists("surveillance_radar_output.csv"):
        return None, None, None, None

    df_radar = pd.read_csv("surveillance_radar_output.csv")
    df_labs = (
        pd.read_csv("dispatched_lab_orders.csv")
        if os.path.exists("dispatched_lab_orders.csv")
        else pd.DataFrame()
    )
    df_sms = (
        pd.read_csv("dispatched_sms_broadcasts.csv")
        if os.path.exists("dispatched_sms_broadcasts.csv")
        else pd.DataFrame()
    )

    briefings = []
    if os.path.exists("incident_command_briefings.json"):
        with open("incident_command_briefings.json", "r", encoding="utf-8") as f:
            briefings = json.load(f)

    return df_radar, df_labs, df_sms, briefings


df_radar, df_labs, df_sms, briefings = load_surveillance_data()

if df_radar is None:
    st.error(
        "Missing pipeline files. Please run Phase 1, Phase 2, Phase 3/4, and Phase 6 first."
    )
    st.stop()

# =====================================================================
# HEADER & TOP METRICS BAR
# =====================================================================
st.title("🐾 Animal-Health Surveillance & Decision Support System")
st.caption(
    "Real-Time Veterinary Outbreak Radar • Spatiotemporal Clustering • Automated Containment"
)

active_outbreaks = df_radar[
    df_radar["cluster_status"] == "ACTIVE_OUTBREAK_SURGE"
]
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Ingested Reports", len(df_radar))
col2.metric("Active Outbreak Clusters", active_outbreaks["cluster_id"].nunique())
col3.metric("Cases in Surge Pockets", len(active_outbreaks))
col4.metric("Lab Orders Dispatched", len(df_labs))
col5.metric("Multilingual SMS Sent", len(df_sms))

st.markdown("---")

# =====================================================================
# SIDEBAR FILTERS
# =====================================================================
st.sidebar.header("🕹️ Surveillance Controls")
selected_status = st.sidebar.multiselect(
    "Filter by Cluster Status:",
    options=df_radar["cluster_status"].unique(),
    default=["ACTIVE_OUTBREAK_SURGE", "ROUTINE_NOISE"],
)

day_range = st.sidebar.slider(
    "Surveillance Timeline Window (Days):",
    min_value=int(df_radar["day"].min()),
    max_value=int(df_radar["day"].max()),
    value=(0, int(df_radar["day"].max())),
)

# Apply filters
filtered_df = df_radar[
    (df_radar["cluster_status"].isin(selected_status))
    & (df_radar["day"] >= day_range[0])
    & (df_radar["day"] <= day_range[1])
]

# =====================================================================
# MAIN PANELS: RADAR MAP & TIMELINE ANOMALIES
# =====================================================================
left_panel, right_panel = st.columns([1.2, 1])

with left_panel:
    st.subheader("📍 Geospatial Surveillance Radar")

    # Streamlit native map projection
    map_df = filtered_df[["latitude", "longitude", "cluster_status"]].copy()
    map_df.dropna(subset=["latitude", "longitude"], inplace=True)
    st.map(map_df, latitude="latitude", longitude="longitude", size=25, zoom=9)

    st.info(
        "💡 **Map Context:** Points denote active reporting nodes across Maharashtra blocks (Pune, Ahmednagar, Kolhapur)."
    )

with right_panel:
    st.subheader("📈 Case Velocity & Temporal Trajectory")

    # Group case volume by day and status
    timeline_data = (
        filtered_df.groupby(["day", "cluster_status"])
        .size()
        .unstack(fill_value=0)
    )
    st.bar_chart(timeline_data, height=360)

# =====================================================================
# INCIDENT COMMAND & ACTIVE OUTBREAK CLUSTERS
# =====================================================================
st.markdown("---")
st.subheader("🚨 Active Outbreak Clusters & Containment Briefings")

if briefings:
    for b in briefings:
        with st.expander(
            f"⚠️ **{b['incident_id']}** — Pathogen: {b['dominant_pathogen']} (Confidence: {b['confidence_score']})",
            expanded=True,
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.write(f"**District:** {b['primary_district']}")
            c2.write(f"**Transmission Velocity:** {b['transmission_velocity']}")
            c3.write(f"**Attack Density:** {b['attack_density']}")
            c4.write(f"**Quarantine Ring:** {b['recommended_quarantine_radius_km']} km")

            st.write(f"**Affected Village Nodes:** `{', '.join(b['affected_villages'])}`")
            st.error(f"**Action Protocol:** {b['action_required']}")
else:
    st.success("No active critical outbreak clusters currently detected.")

# =====================================================================
# CLINICAL CASE INSPECTOR & DISPATCH LOGS
# =====================================================================
st.markdown("---")
tab1, tab2, tab3 = st.tabs(
    ["🔬 Clinical Triage Inspector", "🧪 Dispatched Lab Orders", "📲 Multilingual SMS Queue"]
)

with tab1:
    st.markdown("#### Detailed Case Records & Dynamic Follow-Up Prompts")
    inspector_cols = [
        "report_id",
        "species",
        "symptoms",
        "predicted_disease",
        "diagnostic_confidence_pct",
        "safety_net_alert_level",
        "followup_verification_prompt",
    ]
    st.dataframe(filtered_df[inspector_cols].head(50), use_container_width=True)

with tab2:
    st.markdown("#### Priority Laboratory Sampling Orders (Sentinel Tracking)")
    if not df_labs.empty:
        st.dataframe(
            df_labs[
                [
                    "order_id",
                    "sample_id",
                    "suspected_pathogen",
                    "specimen_required",
                    "transport_protocol",
                    "destination_facility",
                    "dispatch_priority",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.write("No lab orders generated yet.")

with tab3:
    st.markdown("#### Automated Multilingual Alerts (Marathi & English)")
    if not df_sms.empty:
        st.dataframe(
            df_sms[
                [
                    "broadcast_id",
                    "recipient_role",
                    "language",
                    "message_body",
                    "delivery_channel",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.write("No SMS advisories queued.")