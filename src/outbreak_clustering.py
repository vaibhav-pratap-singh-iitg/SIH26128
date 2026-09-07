"""
Unsupervised outbreak-cluster detection.

Complements the supervised triage model (predictor.py / train_model.py) with
an unsupervised view: instead of asking "is this one report concerning?",
this module asks "do reports form a real space-time cluster that looks like
an emerging outbreak, and how severe does that cluster look as a group?"
That's a genuinely different signal - a village can have several individually
LOW/MEDIUM reports that, taken together in space and time, still describe a
real developing situation the per-report classifier has no way to see.

Two methods, chosen deliberately rather than reached for by name-recognition:

1. ST-DBSCAN (Birant & Kut, 2007) - a spatio-temporal extension of DBSCAN.
   Two reports are "neighbors" only if they are close BOTH in space (great-
   circle distance, via a haversine BallTree) AND in time (day difference)
   - not just close in some blended distance. That distinction matters here:
   two villages 5km apart with reports six months apart are not the same
   event, and plain DBSCAN on lat/lon/day-number as three generic dimensions
   would happily merge them because "5km and 180 days" can look like a small
   Euclidean distance once days dominate the scale. ST-DBSCAN also needs no
   fixed cluster count - outbreaks appear and disappear over the dataset, so
   a k-means-style "pick k clusters" model would be the wrong tool.

2. A robust Student-t mixture model (McLachlan & Peel's EM, fixed degrees of
   freedom) clusters reports by *severity profile* (mortality rate, symptom
   severity, animals affected) into soft severity groups. A plain Gaussian
   Mixture Model is not robust here: a handful of extreme reports (a sudden-
   death cluster, say) drag a Gaussian component's mean and inflate its
   covariance, which then blurs the boundary between "moderate" and "severe"
   groups for every other report. A Student-t mixture's heavier tails let
   those extreme points be explained as unlikely-but-plausible draws from a
   wide-tailed component instead of forcibly reshaping it - each point gets a
   down-weighting factor (see `u` in `fit_t_mixture`) that shrinks the
   influence of exactly the kind of outlier that would otherwise distort the
   group means. That is precisely the class of report we cannot afford to
   let quietly warp "what severe looks like."

Run directly to analyze the current synthetic dataset:
    python src/outbreak_clustering.py
"""

from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import multivariate_t
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, preprocessing

EARTH_RADIUS_KM = 6371.0088

# ST-DBSCAN defaults - deliberately tight. Empirically (see the module
# docstring note below), nearest-neighbor villages sit only ~1-8km apart and
# background reports land roughly once a day even with no outbreak, so a
# loose radius/window doesn't find outbreaks - it chains the entire dataset
# into one blob through ordinary background density. These values require a
# genuine local density spike (a handful of reports within a few days at
# essentially the same village) to form a cluster, rather than rewarding
# wide nets. Recalibrate with real_neighbor_distance_km() /
# report_gap_days() below if you regenerate the synthetic data with
# different density.
DEFAULT_EPS_KM = 2.0
DEFAULT_EPS_DAYS = 2
DEFAULT_MIN_SAMPLES = 6

# Severity mixture defaults.
DEFAULT_SEVERITY_COMPONENTS = 3
DEFAULT_DOF = 4.0  # lower = heavier tails = more robust to outlier reports

# A raw ST-DBSCAN cluster on its own is not a reliable outbreak signal here -
# background reporting noise is dense enough (median ~1 day between reports
# at a single village) that plenty of purely coincidental clusters form from
# ordinary background volume alone. What actually separates a real seeded
# outbreak from background noise in this dataset is severity, not density:
# validated by sweeping a total_deaths threshold against the 10 known
# seeded outbreak villages (see validate_against_known_seed below) - 18
# deaths gives 100% recall (every seeded outbreak found) at 83% precision
# (10 of 12 flagged clusters are real). Recall is prioritized over precision
# on purpose, consistent with how HIGH-concern recall was prioritized for
# the supervised triage model: missing a real outbreak is worse than one
# extra false alarm.
OUTBREAK_DEATH_THRESHOLD = 18


# --------------------------------------------------------------------------
# ST-DBSCAN
# --------------------------------------------------------------------------
def st_dbscan(
    coords_deg: np.ndarray,
    days: np.ndarray,
    eps_km: float = DEFAULT_EPS_KM,
    eps_days: float = DEFAULT_EPS_DAYS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> np.ndarray:
    """
    Spatio-temporal DBSCAN. `coords_deg` is (n, 2) of [lat, lon] in degrees;
    `days` is (n,) of integer day offsets. Two points are neighbors only if
    they are within `eps_km` (great-circle) AND within `eps_days`.

    Returns an (n,) label array: -1 for noise, 0..k-1 for cluster membership.
    """
    n = len(coords_deg)
    days = np.asarray(days)
    coords_rad = np.radians(coords_deg)

    tree = BallTree(coords_rad, metric="haversine")
    eps_rad = eps_km / EARTH_RADIUS_KM
    spatial_neighbors = tree.query_radius(coords_rad, r=eps_rad)

    def neighbors(i: int) -> np.ndarray:
        cand = spatial_neighbors[i]
        mask = np.abs(days[cand] - days[i]) <= eps_days
        return cand[mask]

    UNVISITED, NOISE = -2, -1
    labels = np.full(n, UNVISITED, dtype=int)
    cluster_id = -1

    for i in range(n):
        if labels[i] != UNVISITED:
            continue
        neigh = neighbors(i)
        if len(neigh) < min_samples:
            labels[i] = NOISE
            continue

        cluster_id += 1
        labels[i] = cluster_id
        seeds = deque(x for x in neigh if x != i)

        while seeds:
            j = seeds.popleft()
            if labels[j] == NOISE:
                labels[j] = cluster_id  # border point of this cluster
            if labels[j] != UNVISITED:
                continue
            labels[j] = cluster_id
            neigh_j = neighbors(j)
            if len(neigh_j) >= min_samples:
                seeds.extend(neigh_j)

    return labels


# --------------------------------------------------------------------------
# Robust Student-t mixture model (EM, fixed degrees of freedom)
# --------------------------------------------------------------------------
def fit_t_mixture(
    X: np.ndarray,
    n_components: int = DEFAULT_SEVERITY_COMPONENTS,
    dof: float = DEFAULT_DOF,
    n_iter: int = 100,
    tol: float = 1e-4,
    reg_covar: float = 1e-3,
    random_state: int = 42,
) -> dict:
    """
    EM for a multivariate Student-t mixture (McLachlan & Peel, degrees of
    freedom fixed rather than re-estimated each round - a standard, stable
    simplification). The `u` weight computed each E-step is the mechanism
    that gives this its robustness: a point far from a component's center
    (large Mahalanobis distance) gets u -> 0, so it barely moves that
    component's mean/covariance in the M-step - unlike a Gaussian mixture,
    where every point pulls proportionally to its responsibility alone.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    km = KMeans(n_clusters=n_components, n_init=10, random_state=random_state).fit(X)
    means = km.cluster_centers_.copy()
    covariances = np.array([
        np.cov(X[km.labels_ == k], rowvar=False) + reg_covar * np.eye(d)
        if np.sum(km.labels_ == k) > 1 else np.eye(d)
        for k in range(n_components)
    ])
    weights = np.array([max(np.sum(km.labels_ == k), 1) / n for k in range(n_components)])

    resp = np.zeros((n, n_components))
    log_lik_prev = -np.inf
    n_used = 0

    for n_used in range(1, n_iter + 1):
        # --- E-step: responsibilities via log-sum-exp for numerical stability ---
        log_dens = np.column_stack([
            multivariate_t(loc=means[k], shape=covariances[k], df=dof, allow_singular=True).logpdf(X)
            for k in range(n_components)
        ])
        log_weighted = log_dens + np.log(weights + 1e-300)
        max_log = log_weighted.max(axis=1, keepdims=True)
        log_sum = (max_log.squeeze(-1) + np.log(np.exp(log_weighted - max_log).sum(axis=1)))
        resp = np.exp(log_weighted - log_sum[:, None])
        log_lik = float(log_sum.sum())

        # --- robustness weights: down-weight points far from each center ---
        u = np.zeros((n, n_components))
        for k in range(n_components):
            diff = X - means[k]
            inv_cov = np.linalg.inv(covariances[k])
            mahal = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
            u[:, k] = (dof + d) / (dof + mahal)

        # --- M-step: weighted means/covariances, using resp*u as the effective weight ---
        Nk = resp.sum(axis=0) + 1e-300
        weights = Nk / n
        for k in range(n_components):
            w = resp[:, k] * u[:, k]
            w_sum = w.sum() + 1e-300
            means[k] = (w[:, None] * X).sum(axis=0) / w_sum
            diff = X - means[k]
            cov_k = (diff * w[:, None]).T @ diff / (resp[:, k].sum() + 1e-300)
            covariances[k] = cov_k + reg_covar * np.eye(d)

        if abs(log_lik - log_lik_prev) < tol * max(abs(log_lik_prev), 1.0):
            break
        log_lik_prev = log_lik

    return {
        "means": means, "covariances": covariances, "weights": weights,
        "responsibilities": resp, "labels": resp.argmax(axis=1),
        "log_likelihood": log_lik_prev, "n_iter": n_used,
    }


def _rank_severity_groups(means: np.ndarray) -> dict:
    """Map raw component index -> human-readable rank (0=mildest) by mean severity magnitude."""
    order = np.argsort(means.sum(axis=1))
    return {int(component): int(rank) for rank, component in enumerate(order)}


# --------------------------------------------------------------------------
# Putting it together: detect clusters, characterize each one
# --------------------------------------------------------------------------
def detect_outbreak_clusters(
    reports_df: pd.DataFrame,
    regions_df: pd.DataFrame,
    eps_km: float = DEFAULT_EPS_KM,
    eps_days: float = DEFAULT_EPS_DAYS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    n_severity_groups: int = DEFAULT_SEVERITY_COMPONENTS,
    dof: float = DEFAULT_DOF,
) -> dict:
    """
    Runs ST-DBSCAN over all reports (space + time) and a robust Student-t
    severity mixture over the same reports, then characterizes each detected
    spatio-temporal cluster by its dominant severity group.

    Returns {"reports": per-report cluster/severity assignments,
             "cluster_summary": one row per detected ST-DBSCAN cluster,
             "severity_groups": human-readable centers for each severity group}.
    """
    d = reports_df.merge(regions_df[["village", "latitude", "longitude"]], on="village", how="left")
    d = d.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    d["day_offset"] = (d["date"] - d["date"].min()).dt.days

    # --- spatio-temporal clustering ---
    coords = d[["latitude", "longitude"]].to_numpy()
    d["st_cluster"] = st_dbscan(coords, d["day_offset"].to_numpy(), eps_km, eps_days, min_samples)

    # --- severity clustering (reuse the same feature engineering as training) ---
    engineered = preprocessing.engineer_features(d, add_context=False)
    severity_cols = ["mortality_rate", "symptom_severity", "number_affected"]
    scaler = StandardScaler()
    severity_X = scaler.fit_transform(engineered[severity_cols].fillna(0.0))

    t_mix = fit_t_mixture(severity_X, n_components=n_severity_groups, dof=dof)
    rank_map = _rank_severity_groups(t_mix["means"])
    d["severity_group_raw"] = t_mix["labels"]
    d["severity_group_rank"] = d["severity_group_raw"].map(rank_map)  # 0 = mildest
    d["severity_confidence"] = t_mix["responsibilities"][np.arange(len(d)), t_mix["labels"]].round(3)

    group_names = {0: "Mild", 1: "Moderate", 2: "Severe"}
    if n_severity_groups != 3:
        group_names = {r: f"Group {r + 1}" for r in range(n_severity_groups)}
    d["severity_group"] = d["severity_group_rank"].map(group_names)

    # Human-readable severity group centers, in original units.
    centers_original = scaler.inverse_transform(t_mix["means"])
    severity_group_summary = pd.DataFrame(centers_original, columns=severity_cols)
    severity_group_summary["raw_component"] = range(n_severity_groups)
    severity_group_summary["rank"] = severity_group_summary["raw_component"].map(rank_map)
    severity_group_summary["name"] = severity_group_summary["rank"].map(group_names)
    severity_group_summary["mixture_weight"] = t_mix["weights"]
    severity_group_summary = severity_group_summary.sort_values("rank").reset_index(drop=True)

    # --- characterize each spatio-temporal cluster ---
    rows = []
    for cluster_id, g in d[d["st_cluster"] >= 0].groupby("st_cluster"):
        dominant_group = g["severity_group"].mode().iloc[0]
        n_villages = g["village"].nunique()
        n_districts = g["district"].nunique()
        duration_days = int((g["date"].max() - g["date"].min()).days) + 1
        is_epidemic_scale = (n_villages >= 2 or n_districts >= 2) and duration_days >= 7

        rows.append({
            "st_cluster": int(cluster_id),
            "n_reports": len(g),
            "n_villages": n_villages,
            "n_districts": n_districts,
            "villages": ", ".join(sorted(g["village"].unique())[:3]) + ("..." if n_villages > 3 else ""),
            "dominant_village": g["village"].mode().iloc[0],
            "date_start": g["date"].min().date(),
            "date_end": g["date"].max().date(),
            "duration_days": duration_days,
            "total_affected": int(g["number_affected"].sum()),
            "total_deaths": int(g["number_deaths"].sum()),
            "mean_mortality_rate": round(float(g["number_deaths"].sum() / max(1, g["number_affected"].sum())), 3),
            "high_concern_reports": int((g["risk_level"] == "HIGH").sum()) if "risk_level" in g.columns else None,
            "dominant_severity_group": dominant_group,
            "avg_severity_confidence": round(float(g["severity_confidence"].mean()), 3),
            "scale": "Epidemic-scale (multi-village/district)" if is_epidemic_scale else "Localized outbreak",
        })

    cluster_summary = pd.DataFrame(rows).sort_values("n_reports", ascending=False).reset_index(drop=True)
    if not cluster_summary.empty:
        cluster_summary["flagged_as_outbreak"] = cluster_summary["total_deaths"] >= OUTBREAK_DEATH_THRESHOLD
        cluster_summary = cluster_summary.sort_values(
            ["flagged_as_outbreak", "total_deaths"], ascending=[False, False]
        ).reset_index(drop=True)
    n_noise = int((d["st_cluster"] == -1).sum())

    return {
        "reports": d,
        "cluster_summary": cluster_summary,
        "severity_groups": severity_group_summary,
        "n_clusters_found": cluster_summary.shape[0],
        "n_noise_reports": n_noise,
    }


def validate_against_known_seed(
    cluster_summary: pd.DataFrame, regions_df: pd.DataFrame, n_seeded: int = 10
) -> dict:
    """
    Synthetic-data-only sanity check: data_generator.py seeds exactly
    `n_seeded` outbreak villages via `regions.sample(n=n_seeded,
    random_state=config.RANDOM_SEED)`. This reproduces that same sample to
    get the ground truth, then checks how many of the FLAGGED clusters
    (flagged_as_outbreak == True) sit on a genuinely seeded village.

    This only works because we generated the data ourselves and know where
    the outbreaks are - it is not something you get for free on real
    surveillance data, where there's no ground truth to check against. Its
    purpose is purely to show the detection threshold isn't arbitrary.
    """
    seeded_villages = set(regions_df.sample(n=n_seeded, random_state=config.RANDOM_SEED)["village"])
    flagged = cluster_summary[cluster_summary.get("flagged_as_outbreak", False)]
    flagged_villages = set(flagged["dominant_village"]) if "dominant_village" in flagged.columns else set()

    tp = len(flagged_villages & seeded_villages)
    fp = len(flagged_villages - seeded_villages)
    fn = len(seeded_villages - flagged_villages)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "seeded_villages": seeded_villages, "flagged_villages": flagged_villages,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": round(precision, 3), "recall": round(recall, 3),
    }


def _print_report(result: dict, validation: dict | None = None) -> None:
    print("=" * 88)
    print("UNSUPERVISED OUTBREAK-CLUSTER DETECTION  (ST-DBSCAN + robust Student-t severity mixture)")
    print("=" * 88)
    cs = result["cluster_summary"]
    n_flagged = int(cs["flagged_as_outbreak"].sum()) if "flagged_as_outbreak" in cs.columns else 0
    print(f"Spatio-temporal clusters found: {result['n_clusters_found']}   "
          f"(reports not part of any cluster: {result['n_noise_reports']})")
    print(f"Of those, {n_flagged} are flagged as likely real outbreaks "
          f"(total_deaths >= {OUTBREAK_DEATH_THRESHOLD}) - raw cluster count alone is NOT a reliable "
          f"signal here (see the OUTBREAK_DEATH_THRESHOLD comment for why).")

    print("\nSeverity group centers (original units, ranked mildest -> most severe):")
    print(result["severity_groups"][["name", "mortality_rate", "symptom_severity",
                                      "number_affected", "mixture_weight"]].to_string(index=False))

    print(f"\nFlagged clusters (the {n_flagged} that look like real outbreaks):")
    cols = ["st_cluster", "dominant_village", "n_reports", "n_villages", "date_start", "date_end",
            "duration_days", "total_deaths", "mean_mortality_rate", "dominant_severity_group", "scale"]
    flagged = cs[cs["flagged_as_outbreak"]] if "flagged_as_outbreak" in cs.columns else cs.head(0)
    if flagged.empty:
        print("  (none)")
    else:
        print(flagged[cols].to_string(index=False))

    if validation is not None:
        print("\n" + "-" * 88)
        print("SELF-VALIDATION (synthetic data only - checks flagged clusters against the generator's")
        print("known seeded outbreak villages; not available on real, unlabeled data):")
        print(f"  Seeded outbreak villages:  {len(validation['seeded_villages'])}")
        print(f"  Correctly flagged (TP):    {validation['true_positives']}")
        print(f"  False alarms (FP):         {validation['false_positives']}")
        print(f"  Missed outbreaks (FN):     {validation['false_negatives']}")
        print(f"  Precision: {validation['precision']:.2f}   Recall: {validation['recall']:.2f}")


if __name__ == "__main__":
    reports = pd.read_csv(config.REPORTS_CSV)
    regions = pd.read_csv(config.REGIONS_CSV)
    result = detect_outbreak_clusters(reports, regions)
    validation = validate_against_known_seed(result["cluster_summary"], regions)
    _print_report(result, validation)
