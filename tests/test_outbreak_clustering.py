import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.outbreak_clustering import fit_t_mixture, st_dbscan  # noqa: E402


def test_st_dbscan_finds_one_dense_cluster_and_marks_rest_as_noise():
    # 8 points tightly packed in space and time (a real burst), plus a few
    # isolated points scattered far away in both space and time.
    rng = np.random.default_rng(0)
    burst_coords = np.array([[20.0, 74.0]] * 8) + rng.normal(0, 0.001, size=(8, 2))
    burst_days = np.array([10, 11, 11, 12, 12, 13, 13, 14])

    isolated_coords = np.array([[10.0, 60.0], [25.0, 90.0], [5.0, 50.0]])
    isolated_days = np.array([1, 100, 200])

    coords = np.vstack([burst_coords, isolated_coords])
    days = np.concatenate([burst_days, isolated_days])

    labels = st_dbscan(coords, days, eps_km=5.0, eps_days=3, min_samples=5)

    burst_labels = labels[:8]
    isolated_labels = labels[8:]
    assert len(set(burst_labels)) == 1  # all burst points share one cluster
    assert burst_labels[0] != -1  # and it's a real cluster, not noise
    assert all(lbl == -1 for lbl in isolated_labels)  # isolated points are noise


def test_st_dbscan_requires_both_space_and_time_proximity():
    # Two tight spatial clusters at the SAME location, far apart in time -
    # a spatio-temporal method must NOT merge these into one cluster, even
    # though a purely spatial method would.
    rng = np.random.default_rng(1)
    cluster_a = np.array([[20.0, 74.0]] * 6) + rng.normal(0, 0.001, size=(6, 2))
    cluster_b = np.array([[20.0, 74.0]] * 6) + rng.normal(0, 0.001, size=(6, 2))
    coords = np.vstack([cluster_a, cluster_b])
    days = np.concatenate([np.full(6, 5), np.full(6, 500)])  # ~495 days apart

    labels = st_dbscan(coords, days, eps_km=5.0, eps_days=10, min_samples=5)

    labels_a, labels_b = labels[:6], labels[6:]
    assert len(set(labels_a)) == 1 and labels_a[0] != -1
    assert len(set(labels_b)) == 1 and labels_b[0] != -1
    assert labels_a[0] != labels_b[0]  # same place, different time -> different clusters


def test_t_mixture_recovers_well_separated_groups():
    rng = np.random.default_rng(2)
    group_low = rng.normal(loc=[0, 0], scale=0.3, size=(60, 2))
    group_high = rng.normal(loc=[10, 10], scale=0.3, size=(60, 2))
    X = np.vstack([group_low, group_high])

    result = fit_t_mixture(X, n_components=2, dof=4.0, random_state=0)
    labels = result["labels"]

    # Every point in a well-separated group should share one label with its
    # groupmates (label identity is arbitrary, so check purity per group).
    low_labels = labels[:60]
    high_labels = labels[60:]
    assert len(set(low_labels)) == 1
    assert len(set(high_labels)) == 1
    assert low_labels[0] != high_labels[0]


def test_t_mixture_is_robust_to_a_handful_of_extreme_outliers():
    # A Gaussian mixture would let a few extreme outliers drag a component's
    # mean toward them; the t-mixture's heavier tails should keep the main
    # group's fitted mean close to where most of its mass actually is.
    rng = np.random.default_rng(3)
    main_group = rng.normal(loc=[0, 0], scale=0.3, size=(95, 2))
    outliers = np.array([[50.0, 50.0]] * 5)  # a few wild extreme points
    X = np.vstack([main_group, outliers])

    result = fit_t_mixture(X, n_components=1, dof=4.0, random_state=0)
    fitted_mean = result["means"][0]

    # The fitted mean should stay much closer to the main group's true
    # center (0, 0) than to the outliers (50, 50).
    dist_to_main = np.linalg.norm(fitted_mean - np.array([0.0, 0.0]))
    dist_to_outliers = np.linalg.norm(fitted_mean - np.array([50.0, 50.0]))
    assert dist_to_main < dist_to_outliers
    assert dist_to_main < 5.0  # should stay quite close to the true center
