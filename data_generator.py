import json
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fix random seeds for reproducibility
np.random.seed(42)
random.seed(42)


# =====================================================================
# BLOCK 1: DISTRIBUTION FUNCTIONS & CONFIGURATION
# (Change your active distribution and hyper-parameters here)
# =====================================================================
DISTRIBUTION_CONFIG = {
    # Available options: 'poisson', 'neg_binomial', 'gaussian', 'lognormal'
    "active_distribution": "neg_binomial",
    "peak_day": 27.5,
    "duration": 15.0,
    "base_scale": 22.0,  # Peak amplitude factor
    "overdispersion_factor": 3.0,  # Used for neg_binomial (higher = more erratic superspreading)
}


def sample_outbreak_cases(
    day: int,
    peak_day: float = 27.5,
    duration: float = 15.0,
    dist_type: str = "neg_binomial",
    base_scale: float = 22.0,
    overdispersion_factor: float = 3.0,
) -> int:
    """Computes daily injected surge cases based on chosen probability distribution.

    Options: 'poisson', 'neg_binomial', 'gaussian', 'lognormal'
    """
    t_dist = abs(day - peak_day)
    half_width = duration / 2.0

    if t_dist > half_width:
        return 0

    # Normalized progression factor (1.0 at peak, 0.0 at edges)
    envelope = max(0.0, 1.0 - (t_dist / half_width))

    if dist_type == "poisson":
        lam = max(0.5, base_scale * envelope)
        return int(np.random.poisson(lam=lam))

    elif dist_type == "neg_binomial":
        # Superspreader dynamic (variance > mean)
        mean = max(1.0, base_scale * envelope)
        variance = mean + (mean**2) / max(0.1, overdispersion_factor)
        p = mean / variance
        n = (mean**2) / (variance - mean)
        return int(np.random.negative_binomial(n=max(1, int(n)), p=p))

    elif dist_type == "gaussian":
        sigma = duration / 4.0
        intensity = base_scale * np.exp(
            -0.5 * ((day - peak_day) / max(0.1, sigma)) ** 2
        )
        return int(
            max(
                0,
                np.random.normal(
                    loc=intensity, scale=np.sqrt(intensity) + 1e-3
                ),
            )
        )

    elif dist_type == "lognormal":
        # Asymmetric right-skewed tail
        window_start = peak_day - (duration * 0.3)
        delta_t = max(0.1, day - window_start)
        log_intensity = (
            base_scale * (delta_t**1.5) * np.exp(-delta_t / 3.0) / 4.0
        )
        return int(np.random.poisson(lam=max(0.5, log_intensity)))

    else:
        raise ValueError(
            f"Unsupported distribution: '{dist_type}'. Choose from 'poisson', 'neg_binomial', 'gaussian', 'lognormal'."
        )


# =====================================================================
# BLOCK 2: RELATIONAL DATASET GENERATOR
# =====================================================================
def generate_surveillance_dataset():
    """Generates all relational tables and injects the outbreak using the active distribution."""
    # 1. Geographic Hierarchy (Maharashtra Anchor)
    districts_data = {
        "Pune": {
            "Haveli": ["Khadakwasla", "Wagholi", "Uruli Kanchan"],
            "Baramati": ["Malegaon Bk", "Supa", "Deulgaon Rasal"],
            "Shirur": ["Shikrapur", "Pabal", "Talegaon Dhamdhere"],
        },
        "Ahmednagar": {
            "Rahuri": ["Vambori", "Devlali Pravara", "Taharababad"],
            "Sangamner": ["Ashwi", "Sakur", "Talegaon"],
        },
        "Kolhapur": {
            "Karvir": ["Ujalaiwadi", "Koparde", "Vadanange"],
            "Shirol": ["Jaysingpur", "Kurundwad", "Alas"],
        },
    }

    village_rows = []
    v_counter = 101
    base_lat, base_lon = 18.5204, 73.8567

    for dist, blocks in districts_data.items():
        for blk, villages in blocks.items():
            for vil in villages:
                village_rows.append(
                    {
                        "village_id": f"VIL_{v_counter}",
                        "village_name": vil,
                        "block_name": blk,
                        "district_name": dist,
                        "latitude": round(
                            base_lat + np.random.uniform(-1.2, 1.2), 5
                        ),
                        "longitude": round(
                            base_lon + np.random.uniform(-0.8, 1.5), 5
                        ),
                        "cattle_pop": np.random.randint(400, 2200),
                        "buffalo_pop": np.random.randint(200, 1500),
                        "goat_pop": np.random.randint(500, 3000),
                        "sheep_pop": np.random.randint(100, 1200),
                        "poultry_pop": np.random.randint(1000, 10000),
                    }
                )
                v_counter += 1

    df_geo = pd.DataFrame(village_rows)

    # 2. Disease Taxonomy
    disease_taxonomy = {
        "Foot_and_Mouth_Disease": {
            "target_species": ["cattle", "buffalo", "sheep", "goat"],
            "cardinal_symptoms": [
                "high_fever",
                "oral_blisters",
                "foot_lesions",
                "excessive_salivation",
                "lameness",
            ],
            "severity_weight": "high",
        },
        "Lumpy_Skin_Disease": {
            "target_species": ["cattle", "buffalo"],
            "cardinal_symptoms": [
                "cutaneous_nodules",
                "fever",
                "enlarged_lymph_nodes",
                "edema_in_legs",
                "nasal_discharge",
            ],
            "severity_weight": "critical",
        },
        "PPR_Goat_Plague": {
            "target_species": ["goat", "sheep"],
            "cardinal_symptoms": [
                "sudden_fever",
                "erosive_stomatitis",
                "profuse_diarrhea",
                "respiratory_distress",
            ],
            "severity_weight": "critical",
        },
        "Avian_Influenza": {
            "target_species": ["poultry"],
            "cardinal_symptoms": [
                "cyanosis_comb_wattle",
                "facial_edema",
                "watery_diarrhea",
                "abrupt_drop_egg_production",
            ],
            "severity_weight": "critical",
        },
        "Anthrax": {
            "target_species": ["cattle", "buffalo", "sheep", "goat"],
            "cardinal_symptoms": [
                "sudden_death",
                "unclotted_blood_orifices",
                "apoplexy",
                "colic",
            ],
            "severity_weight": "emergency",
        },
    }

    all_noise_symptoms = [
        "mild_anorexia",
        "sluggishness",
        "minor_cough",
        "eye_watering",
        "transient_indigestion",
        "tick_infestation",
    ]

    # 3. User / Actor Profiles
    user_rows = []
    u_counter = 1001

    for _, v in df_geo.iterrows():
        for _ in range(8):  # 8 Farmers per village
            user_rows.append(
                {
                    "user_id": f"USR_{u_counter}",
                    "role": "farmer",
                    "village_id": v["village_id"],
                    "preferred_language": np.random.choice(
                        ["Marathi", "English", "Hindi"], p=[0.80, 0.10, 0.10]
                    ),
                    "cattle_count": np.random.randint(1, 8),
                    "buffalo_count": np.random.randint(0, 5),
                    "goat_count": np.random.randint(0, 15),
                    "sheep_count": np.random.randint(0, 10),
                    "poultry_count": np.random.randint(0, 50),
                }
            )
            u_counter += 1

        # 1 Para-vet per village
        user_rows.append(
            {
                "user_id": f"USR_{u_counter}",
                "role": "para_vet",
                "village_id": v["village_id"],
                "preferred_language": "Marathi",
                "cattle_count": 0,
                "buffalo_count": 0,
                "goat_count": 0,
                "sheep_count": 0,
                "poultry_count": 0,
            }
        )
        u_counter += 1

    # Vet officers per district
    for dist in df_geo["district_name"].unique():
        user_rows.append(
            {
                "user_id": f"USR_{u_counter}",
                "role": "vet_officer",
                "village_id": df_geo[df_geo["district_name"] == dist][
                    "village_id"
                ].iloc[0],
                "preferred_language": "English",
                "cattle_count": 0,
                "buffalo_count": 0,
                "goat_count": 0,
                "sheep_count": 0,
                "poultry_count": 0,
            }
        )
        u_counter += 1

    df_users = pd.DataFrame(user_rows)

    # 4. Reports Engine (Driven by Selected Distribution)
    start_date = datetime(2026, 7, 1)
    total_days = 60
    reports = []
    rep_id = 50001

    cluster_villages = df_geo[
        (df_geo["district_name"] == "Pune")
        & (df_geo["block_name"] == "Haveli")
    ]["village_id"].tolist()

    for day in range(total_days):
        current_date = start_date + timedelta(
            days=day, hours=np.random.randint(6, 20)
        )

        # Baseline Noise
        daily_baseline = np.random.randint(15, 25)
        for _ in range(daily_baseline):
            vil = df_geo.sample(1).iloc[0]
            rep_type = np.random.choice(["farmer", "para_vet"], p=[0.75, 0.25])
            reporter = (
                df_users[
                    (df_users["village_id"] == vil["village_id"])
                    & (df_users["role"] == rep_type)
                ]
                .sample(1)
                .iloc[0]
            )

            is_mortality = np.random.choice([False, True], p=[0.88, 0.12])
            species = np.random.choice(
                ["cattle", "buffalo", "goat", "sheep", "poultry"]
            )
            symptoms = list(
                np.random.choice(
                    all_noise_symptoms, size=np.random.randint(1, 3)
                )
            )

            reports.append(
                {
                    "report_id": f"REP_{rep_id}",
                    "day": day,
                    "timestamp": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "village_id": vil["village_id"],
                    "reporter_id": reporter["user_id"],
                    "reporter_role": rep_type,
                    "species": species,
                    "symptoms": ",".join(symptoms),
                    "severity_score": round(np.random.uniform(1.0, 4.0), 2),
                    "report_type": (
                        "mortality_alert" if is_mortality else "symptom_report"
                    ),
                    "mortality_count": (
                        np.random.randint(1, 2) if is_mortality else 0
                    ),
                    "suspected_disease": "Inconclusive",
                    "is_outbreak_cluster": 0,
                }
            )
            rep_id += 1

        # Outbreak Surge (sampled directly from the active distribution)
        surge_size = sample_outbreak_cases(
            day=day,
            peak_day=DISTRIBUTION_CONFIG["peak_day"],
            duration=DISTRIBUTION_CONFIG["duration"],
            dist_type=DISTRIBUTION_CONFIG["active_distribution"],
            base_scale=DISTRIBUTION_CONFIG["base_scale"],
            overdispersion_factor=DISTRIBUTION_CONFIG["overdispersion_factor"],
        )

        for _ in range(surge_size):
            target_vil = random.choice(cluster_villages)
            reporter = (
                df_users[df_users["village_id"] == target_vil].sample(1).iloc[0]
            )
            lsd_symptoms = list(
                np.random.choice(
                    disease_taxonomy["Lumpy_Skin_Disease"]["cardinal_symptoms"],
                    size=np.random.randint(2, 5),
                    replace=False,
                )
            )
            is_mortality = np.random.choice([False, True], p=[0.82, 0.18])

            reports.append(
                {
                    "report_id": f"REP_{rep_id}",
                    "day": day,
                    "timestamp": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "village_id": target_vil,
                    "reporter_id": reporter["user_id"],
                    "reporter_role": reporter["role"],
                    "species": np.random.choice(
                        ["cattle", "buffalo"], p=[0.85, 0.15]
                    ),
                    "symptoms": ",".join(lsd_symptoms),
                    "severity_score": round(np.random.uniform(7.0, 9.8), 2),
                    "report_type": (
                        "mortality_alert" if is_mortality else "symptom_report"
                    ),
                    "mortality_count": (
                        np.random.randint(1, 4) if is_mortality else 0
                    ),
                    "suspected_disease": "Lumpy_Skin_Disease",
                    "is_outbreak_cluster": 1,
                }
            )
            rep_id += 1

    df_reports = pd.DataFrame(reports)

    # 5. Lab Referrals & Escalations
    lab_records = []
    sample_id = 90001
    high_risk_cases = df_reports[df_reports["severity_score"] > 6.0]

    for _, case in high_risk_cases.iterrows():
        if np.random.rand() < 0.65:
            lab_status = np.random.choice(
                ["Confirmed_Positive", "Negative", "Pending"],
                p=[0.60, 0.25, 0.15],
            )
            sent_dt = datetime.strptime(case["timestamp"], "%Y-%m-%d %H:%M:%S")
            lab_records.append(
                {
                    "sample_id": f"SMP_{sample_id}",
                    "report_id": case["report_id"],
                    "village_id": case["village_id"],
                    "sample_type": np.random.choice(
                        ["Blood", "Nasal_Swab", "Skin_Scraping"]
                    ),
                    "assigned_facility": "District_Diagnostic_Lab_Pune",
                    "dispatch_timestamp": (
                        sent_dt + timedelta(hours=np.random.randint(2, 12))
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "diagnostic_result": lab_status,
                    "escalation_level": (
                        "State_Level_Alert"
                        if lab_status == "Confirmed_Positive"
                        else "Local_Containment"
                    ),
                    "turnaround_hours": (
                        np.random.randint(24, 72)
                        if lab_status != "Pending"
                        else None
                    ),
                }
            )
            sample_id += 1

    df_labs = pd.DataFrame(lab_records)

    # 6. Vaccination Ledger
    vax_records = []
    vx_counter = 70001
    for _, user in df_users[df_users["role"] == "farmer"].sample(150).iterrows():
        for sp, count in [
            ("cattle", user["cattle_count"]),
            ("goat", user["goat_count"]),
        ]:
            if count > 0:
                for idx in range(min(count, 2)):
                    tag = f"IND_{user['user_id']}_{sp[:3].upper()}_{idx+1}"
                    dose_date = start_date - timedelta(
                        days=np.random.randint(30, 280)
                    )
                    vax_records.append(
                        {
                            "vaccination_id": f"VAX_{vx_counter}",
                            "animal_tag_id": tag,
                            "species": sp,
                            "village_id": user["village_id"],
                            "vaccine_type": (
                                "LSD_Attenuated"
                                if sp == "cattle"
                                else "PPR_Live_Attenuated"
                            ),
                            "administered_date": dose_date.strftime("%Y-%m-%d"),
                            "next_booster_due": (
                                dose_date + timedelta(days=365)
                            ).strftime("%Y-%m-%d"),
                            "status": "Protected",
                        }
                    )
                    vx_counter += 1

    df_vax = pd.DataFrame(vax_records)

    # 7. Multilingual Advisory Templates
    advisories = {
        "Lumpy_Skin_Disease": {
            "priority": "CRITICAL",
            "advisory_en": "Outbreak Alert: Lumpy Skin Disease reported in your area. Isolate cattle with skin nodules and restrict animal movement.",
            "advisory_mr": "उद्रेक चेतावणी: आपल्या भागात लंपी त्वचा रोगाचा प्रादुर्भाव आढळला आहे. गाठी दिसणाऱ्या जनावरांना त्वरित वेगळे करा आणि जनावरांची वाहतूक थांबवा.",
        },
        "Foot_and_Mouth_Disease": {
            "priority": "HIGH",
            "advisory_en": "Surveillance Warning: Suspected FMD cases nearby. Disinfect shed entrances with 4% sodium carbonate.",
            "advisory_mr": "दक्षता सूचना: जवळच्या परिसरात लाळ्या खुरकूत (FMD) संशयित लक्षणे आढळली आहेत. गोठ्याचे प्रवेशद्वार निर्जंतुक करा.",
        },
    }

    # Save outputs to disk
    df_geo.to_csv("geo_hierarchy.csv", index=False)
    df_users.to_csv("users.csv", index=False)
    df_reports.to_csv("symptom_mortality_reports.csv", index=False)
    df_labs.to_csv("lab_escalations.csv", index=False)
    df_vax.to_csv("vaccination_records.csv", index=False)

    with open("multilingual_advisories.json", "w", encoding="utf-8") as f:
        json.dump(advisories, f, ensure_ascii=False, indent=2)

    return df_reports, df_geo


# =====================================================================
# BLOCK 3: VISUALIZATION CALL
# =====================================================================
def plot_surveillance_dynamics(df_reports=None, df_geo=None):
    """Visualizes the generated report time series alongside the susceptible vs. infected herd dynamics."""
    # If DataFrames are not passed, load directly from the saved CSVs
    if df_reports is None:
        df_reports = pd.read_csv("symptom_mortality_reports.csv")
    if df_geo is None:
        df_geo = pd.read_csv("geo_hierarchy.csv")

    # Determine targeted outbreak cluster population
    haveli_villages = df_geo[
        (df_geo["district_name"] == "Pune")
        & (df_geo["block_name"] == "Haveli")
    ]["village_id"].tolist()
    total_cluster_pop = df_geo[df_geo["village_id"].isin(haveli_villages)][
        ["cattle_pop", "buffalo_pop"]
    ].values.sum()

    # Aggregate daily metrics
    days = sorted(df_reports["day"].unique())
    daily_noise = []
    daily_cluster = []
    daily_mortality = []

    for d in days:
        sub = df_reports[df_reports["day"] == d]
        daily_noise.append(len(sub[sub["is_outbreak_cluster"] == 0]))
        daily_cluster.append(len(sub[sub["is_outbreak_cluster"] == 1]))
        daily_mortality.append(sub["mortality_count"].sum())

    # Epidemic trajectory accounting
    recovery_window = 10
    active_queue = []
    healthy_susceptible = []
    active_infected = []
    cumulative_recovered = []
    cumulative_deaths = []

    current_susceptible = total_cluster_pop
    rec_count = 0
    death_count = 0

    for d in days:
        new_cluster_cases = daily_cluster[d]
        new_deaths = daily_mortality[d]

        # Shift from susceptible
        infections = min(current_susceptible, new_cluster_cases)
        current_susceptible -= infections

        active_queue.append(infections)
        if len(active_queue) > recovery_window:
            resolving = active_queue.pop(0)
            rec = max(0, resolving - int(resolving * 0.08))
            rec_count += rec

        death_count += new_deaths
        active_infected.append(sum(active_queue))
        healthy_susceptible.append(current_susceptible)
        cumulative_recovered.append(rec_count)
        cumulative_deaths.append(death_count)

    # Render Dual Plot
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]}
    )
    dist_name = DISTRIBUTION_CONFIG["active_distribution"].upper()

    # Top Plot: Case Ingestion Breakdown
    ax1.bar(
        days,
        daily_noise,
        color="#6c757d",
        alpha=0.45,
        label="Background Endemic Noise",
    )
    ax1.bar(
        days,
        daily_cluster,
        bottom=daily_noise,
        color="#d9534f",
        alpha=0.85,
        label=f"Injected Surge ({dist_name})",
    )
    ax1.plot(
        days,
        active_infected,
        color="#0275d8",
        linewidth=2.4,
        label="Active Clinical Load",
    )
    ax1.step(
        days,
        cumulative_deaths,
        color="#292b2c",
        linestyle="--",
        linewidth=1.8,
        label="Cumulative Mortalities",
    )
    ax1.set_ylabel("Daily Report Volume", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"Surveillance Pipeline Validation: Outbreak Ingestion Dynamics [Engine: {dist_name}]",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Bottom Plot: Herd Population Trajectory
    ax2.plot(
        days,
        healthy_susceptible,
        color="#5cb85c",
        linewidth=2.4,
        label="Susceptible Herd Stock",
    )
    ax2.plot(
        days,
        cumulative_recovered,
        color="#17a2b8",
        linewidth=2.0,
        linestyle="-.",
        label="Recovered / Cleared",
    )
    ax2.set_ylabel("Livestock Count", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Surveillance Timeline (Days)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="center left")

    plt.tight_layout()
    plt.show()


# =====================================================================
# EXECUTION
# =====================================================================
if __name__ == "__main__":
    # 1. Run generation pipeline
    df_reports, df_geo = generate_surveillance_dataset()
    print("Files successfully generated:")
    print(" - geo_hierarchy.csv")
    print(" - users.csv")
    print(f" - symptom_mortality_reports.csv ({len(df_reports)} rows)")
    print(" - lab_escalations.csv")
    print(" - vaccination_records.csv")
    print(" - multilingual_advisories.json")

    # 2. Simple function call to visualize
    plot_surveillance_dynamics(df_reports, df_geo)