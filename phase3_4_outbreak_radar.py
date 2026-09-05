from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

# =====================================================================
# MODIFIABLE HYPERPARAMETERS (Tune your surveillance radar here)
# =====================================================================
RADAR_CONFIG = {
    # Spatiotemporal neighborhood envelope
    "eps_spatial_km": 15.0,  # Max distance (km) between linked farm cases
    "eps_temporal_days": 4.0,  # Max time window (days) between linked cases
    "min_pts": 3,  # Minimum connected cases needed to instantiate a cluster
    # Anomaly & Velocity thresholds
    "endemic_velocity_threshold": 1.0,  # Cases/day floor to elevate endemic to active outbreak
    "attack_density_threshold": 2.5,  # Cases per 1,000 livestock to trigger active surge
    # Noise suppression guardrails
    "exclude_noise_from_clustering": True,  # Disallow "Inconclusive / Routine Noise" from seeding clusters
    "min_severity_to_seed": 4.0,  # Minimum clinical severity score to act as a cluster seed
    # Evaluator configuration
    # What cluster classifications count as a positive detection in the safety evaluation:
    "positive_eval_classes": ["ACTIVE_OUTBREAK_SURGE"],
}


# =====================================================================
# 1. REFINED ST-DBSCAN CLUSTERING ENGINE
# =====================================================================
class HighSensitivityST_DBSCAN:
    """Spatiotemporal Density-Based Clustering tailored for veterinary surveillance.

    Filters benign background noise from seeding clusters while maintaining high
    sensitivity to genuine clinical surges.
    """

    def __init__(self, config: dict):
        self.eps_spatial = config["eps_spatial_km"]
        self.eps_temporal = config["eps_temporal_days"]
        self.min_pts = config["min_pts"]
        self.velocity_threshold = config["endemic_velocity_threshold"]
        self.attack_threshold = config["attack_density_threshold"]
        self.exclude_noise = config["exclude_noise_from_clustering"]
        self.min_severity = config["min_severity_to_seed"]

    def _get_neighbors(self, index: int, df: pd.DataFrame) -> list:
        target = df.iloc[index]

        # GUARDRAIL 1: Never allow benign noise or sub-threshold cases to seed clusters
        if self.exclude_noise:
            if target["predicted_disease"] == "Inconclusive / Routine Noise":
                return []
            if target["safety_net_alert_level"] == "ROUTINE_GREEN":
                return []
            if target["severity_score"] < self.min_severity:
                return []

        # 1. Spatial Euclidean distance constraint (km)
        dx = df["x_km"] - target["x_km"]
        dy = df["y_km"] - target["y_km"]
        spatial_dist = np.sqrt(dx**2 + dy**2)

        # 2. Temporal interval constraint (days)
        dt = np.abs(df["delta_t_days"] - target["delta_t_days"])

        # GUARDRAIL 2: Neighbor matching must align on clinical pathogen
        # Disallow matching against generic inconclusive entries
        matching_pathogen = (
            df["predicted_disease"] == target["predicted_disease"]
        ) & (df["predicted_disease"] != "Inconclusive / Routine Noise")

        neighbors = df[
            (spatial_dist <= self.eps_spatial)
            & (dt <= self.eps_temporal)
            & matching_pathogen
        ].index.tolist()

        return neighbors

    def fit_predict(self, df_input: pd.DataFrame) -> pd.DataFrame:
        df = df_input.copy().reset_index(drop=True)
        n = len(df)
        cluster_labels = np.full(n, -1, dtype=int)  # -1 = Routine Noise
        visited = np.full(n, False, dtype=bool)
        current_cluster_id = 0

        # Run density-connectivity discovery
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = self._get_neighbors(i, df)

            if len(neighbors) >= self.min_pts:
                cluster_labels[i] = current_cluster_id
                queue = [idx for idx in neighbors if idx != i]

                while queue:
                    neighbor_idx = queue.pop(0)
                    if not visited[neighbor_idx]:
                        visited[neighbor_idx] = True
                        sub_neighbors = self._get_neighbors(neighbor_idx, df)
                        if len(sub_neighbors) >= self.min_pts:
                            queue.extend(
                                [sn for sn in sub_neighbors if sn not in queue]
                            )

                    if cluster_labels[neighbor_idx] == -1:
                        cluster_labels[neighbor_idx] = current_cluster_id

                current_cluster_id += 1

        df["cluster_id"] = cluster_labels

        # Calculate dynamics per cluster
        df["cluster_status"] = "ROUTINE_NOISE"
        df["outbreak_confidence_index"] = 0.0
        df["transmission_velocity"] = 0.0
        df["attack_density_per_1k"] = 0.0

        for c_id in range(current_cluster_id):
            c_mask = df["cluster_id"] == c_id
            c_data = df[c_mask]
            c_size = len(c_data)

            # Transmission velocity & attack density
            duration = max(
                1.0,
                c_data["delta_t_days"].max() - c_data["delta_t_days"].min(),
            )
            velocity = round(c_size / duration, 2)
            total_census = max(100, c_data["species_census"].unique().sum())
            attack_density = round((c_size / total_census) * 1000.0, 2)

            # Mean diagnostic certainty across cluster cases
            avg_diag_conf = c_data["diagnostic_confidence_pct"].mean() / 100.0

            # Composite Outbreak Confidence Index
            raw_intensity = (velocity * 0.5) + (attack_density * 0.5)
            confidence_index = round(
                float(
                    (2.0 / (1.0 + np.exp(-raw_intensity)) - 1.0)
                    * avg_diag_conf
                    * 100.0
                ),
                1,
            )

            # Differentiate endemic baseline vs active outbreak surge
            if (
                velocity >= self.velocity_threshold
                or attack_density >= self.attack_threshold
            ):
                status = "ACTIVE_OUTBREAK_SURGE"
            else:
                status = "ENDEMIC_CIRCULATION"

            df.loc[c_mask, "cluster_status"] = status
            df.loc[c_mask, "outbreak_confidence_index"] = confidence_index
            df.loc[c_mask, "transmission_velocity"] = velocity
            df.loc[c_mask, "attack_density_per_1k"] = attack_density

        return df


# =====================================================================
# 2. SAFETY-NET EVALUATOR
# =====================================================================
class SafetyNetEvaluator:

    @staticmethod
    def evaluate(df_results: pd.DataFrame, positive_classes: list) -> dict:
        y_true = df_results["is_outbreak_cluster"].values
        y_pred = (df_results["cluster_status"].isin(positive_classes)).astype(int).values

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f2 = (
            (5 * precision * recall) / (4 * precision + recall)
            if (4 * precision + recall) > 0
            else 0.0
        )
        overall_acc = (tp + tn) / len(y_true)

        return {
            "Total Reports": len(y_true),
            "True Positives (Outbreaks Caught)": int(tp),
            "False Negatives (Outbreaks Missed)": int(fn),
            "False Positives (Safety Alarms)": int(fp),
            "True Negatives (Routine Noise Cleared)": int(tn),
            "Recall (Sensitivity)": f"{recall * 100:.2f}%",
            "False Negative Rate (FNR)": f"{fnr * 100:.2f}%",
            "Precision": f"{precision * 100:.2f}%",
            "F2-Score (Safety-Weighted)": round(f2, 4),
            "Overall Accuracy": f"{overall_acc * 100:.2f}%",
        }


# =====================================================================
# 3. DUAL-PANE VISUALIZATION RADAR
# =====================================================================
def plot_outbreak_radar(df_clustered: pd.DataFrame):
    fig, (ax_map, ax_time) = plt.subplots(1, 2, figsize=(15, 6.5))

    # --- LEFT PANEL: Spatial Grid ---
    noise = df_clustered[df_clustered["cluster_id"] == -1]
    ax_map.scatter(
        noise["x_km"],
        noise["y_km"],
        c="#adb5bd",
        alpha=0.35,
        s=20,
        label="Routine Noise (Cleared)",
    )

    clusters = [cid for cid in df_clustered["cluster_id"].unique() if cid != -1]
    palette = ["#d9534f", "#0275d8", "#6f42c1", "#20c997", "#fd7e14"]

    for idx, cid in enumerate(clusters):
        c_pts = df_clustered[df_clustered["cluster_id"] == cid]
        status = c_pts["cluster_status"].iloc[0]
        disease = c_pts["predicted_disease"].iloc[0]
        conf = c_pts["outbreak_confidence_index"].iloc[0]

        color = (
            palette[idx % len(palette)]
            if status == "ACTIVE_OUTBREAK_SURGE"
            else "#f0ad4e"
        )

        ax_map.scatter(
            c_pts["x_km"],
            c_pts["y_km"],
            color=color,
            s=60,
            edgecolors="black",
            linewidth=0.8,
            label=f"Cluster #{cid}: {disease} ({status[:10]}... Conf: {conf}%)",
        )

        # Draw boundary hull around spatial clusters
        if len(c_pts) >= 3:
            coords = c_pts[["x_km", "y_km"]].values
            try:
                hull = ConvexHull(coords)
                for simplex in hull.simplices:
                    ax_map.plot(
                        coords[simplex, 0],
                        coords[simplex, 1],
                        color=color,
                        linestyle="--",
                        linewidth=1.6,
                    )
            except Exception:
                pass

    ax_map.set_title(
        "Geospatial Outbreak Radar (Clusters & Containment Polygons)",
        fontsize=11,
        fontweight="bold",
    )
    ax_map.set_xlabel("Planar Coordinate X (km)", fontsize=10)
    ax_map.set_ylabel("Planar Coordinate Y (km)", fontsize=10)
    ax_map.grid(True, linestyle=":", alpha=0.6)
    ax_map.legend(loc="upper right", fontsize=8)

    # --- RIGHT PANEL: Temporal Transmission Timeline ---
    days = sorted(df_clustered["day"].unique())
    daily_noise = [
        len(
            df_clustered[
                (df_clustered["day"] == d)
                & (df_clustered["cluster_id"] == -1)
            ]
        )
        for d in days
    ]
    daily_endemic = [
        len(
            df_clustered[
                (df_clustered["day"] == d)
                & (df_clustered["cluster_status"] == "ENDEMIC_CIRCULATION")
            ]
        )
        for d in days
    ]
    daily_outbreak = [
        len(
            df_clustered[
                (df_clustered["day"] == d)
                & (
                    df_clustered["cluster_status"]
                    == "ACTIVE_OUTBREAK_SURGE"
                )
            ]
        )
        for d in days
    ]

    ax_time.bar(
        days, daily_noise, color="#adb5bd", alpha=0.45, label="Filtered Routine Noise"
    )
    ax_time.bar(
        days,
        daily_endemic,
        bottom=daily_noise,
        color="#f0ad4e",
        alpha=0.85,
        label="Endemic Baseline Circulation",
    )
    ax_time.bar(
        days,
        daily_outbreak,
        bottom=np.array(daily_noise) + np.array(daily_endemic),
        color="#d9534f",
        alpha=0.9,
        label="Active Outbreak Surge",
    )

    ax_time.set_title(
        "Timeline Anomaly Tracking (Case Ingestion vs Surge Detection)",
        fontsize=11,
        fontweight="bold",
    )
    ax_time.set_xlabel("Surveillance Day", fontsize=10)
    ax_time.set_ylabel("Report Volume", fontsize=10)
    ax_time.grid(True, linestyle=":", alpha=0.6)
    ax_time.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.show()


# =====================================================================
# 4. EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    try:
        df_diagnosed = pd.read_csv("diagnosed_clinical_feed.csv")
    except FileNotFoundError:
        raise FileNotFoundError(
            "Missing 'diagnosed_clinical_feed.csv'. Please run 'phase2_matcher.py' first."
        )

    # 1. Initialize and execute radar
    radar = HighSensitivityST_DBSCAN(RADAR_CONFIG)
    df_clustered = radar.fit_predict(df_diagnosed)
    df_clustered.to_csv("surveillance_radar_output.csv", index=False)

    # 2. Evaluate performance against injected ground truth
    metrics = SafetyNetEvaluator.evaluate(
        df_clustered, RADAR_CONFIG["positive_eval_classes"]
    )

    print("Phase 3 & 4 Radar Execution Complete:")
    print("=" * 48)
    for k, v in metrics.items():
        print(f" - {k:<38}: {v}")
    print("=" * 48)

    # 3. Inspect detected outbreak clusters
    outbreaks = df_clustered[
        df_clustered["cluster_status"] == "ACTIVE_OUTBREAK_SURGE"
    ]
    if not outbreaks.empty:
        for cid in outbreaks["cluster_id"].unique():
            c_data = outbreaks[outbreaks["cluster_id"] == cid]
            print(f"\n[!] Detected Active Outbreak (Cluster #{cid}):")
            print(f" - Flagged Cases       : {len(c_data)}")
            print(f" - Dominant Pathogen   : {c_data['predicted_disease'].iloc[0]}")
            print(f" - Outbreak Confidence : {c_data['outbreak_confidence_index'].iloc[0]}%")
            print(f" - Velocity            : {c_data['transmission_velocity'].iloc[0]} cases/day")
            print(f" - Attack Density      : {c_data['attack_density_per_1k'].iloc[0]} per 1,000 head")

    # 4. Render visual dashboard
    plot_outbreak_radar(df_clustered)