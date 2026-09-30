import numpy as np
import pytest

from src.conformal import LocalisationConformal, conformal_quantile, crc_threshold, weighted_quantile
from src.geometry import bev_iou_matrix, in_roi, points_in_boxes, rotated_nms_bev
from src.metrics import ap_from_matches, average_precision_center, match_frame


def test_bev_iou_identity_and_disjoint():
    b = np.array([[10, 0, 0, 4, 2, 1.5, 0.3]])
    assert bev_iou_matrix(b, b)[0, 0] == pytest.approx(1.0)
    far = b.copy()
    far[0, 0] += 20
    assert bev_iou_matrix(b, far)[0, 0] == 0.0


def test_bev_iou_rotation_invariance_of_pi():
    a = np.array([[5, 5, 0, 4, 2, 1.5, 0.4]])
    b = a.copy()
    b[0, 6] += np.pi
    assert bev_iou_matrix(a, b)[0, 0] == pytest.approx(1.0)


def test_points_in_boxes_rotated():
    box = np.array([[0, 0, 0, 4, 1, 2, np.pi / 2]])  # long axis along y
    pts = np.array([[0, 1.8, 0], [1.8, 0, 0]])
    m = points_in_boxes(pts, box)
    assert m[0, 0] and not m[0, 1]


def test_rotated_nms_keeps_best():
    boxes = np.array([[0, 0, 0, 4, 2, 1.5, 0], [0.2, 0, 0, 4, 2, 1.5, 0], [10, 0, 0, 4, 2, 1.5, 0]])
    keep = rotated_nms_bev(boxes, np.array([0.5, 0.9, 0.7]), 0.1)
    assert list(keep) == [1, 2]


def test_roi():
    roi = {"x_min": 0, "x_max": 69, "y_abs_max": 39.5, "half_fov_deg": 40}
    xy = np.array([[10, 0], [-5, 0], [10, 20], [80, 0]])
    assert list(in_roi(xy, roi)) == [True, False, False, False]


def test_ap_perfect_and_empty():
    gt = [np.array([[10, 0], [20, 5]])]
    assert average_precision_center([(gt[0], np.array([0.9, 0.8]))], gt, 1.0) == pytest.approx(1.0)
    assert average_precision_center([(np.zeros((0, 2)), np.zeros(0))], gt, 1.0) == 0.0


def test_ap_ranking_matters():
    s = np.array([0.9, 0.8, 0.7, 0.6])
    good = ap_from_matches(s, np.array([1, 1, 0, 0], bool), 2)
    bad = ap_from_matches(s, np.array([0, 0, 1, 1], bool), 2)
    assert good > bad


def test_dontcare_ignored():
    tp, ign, _ = match_frame(np.array([[0, 0.0], [30, 0]]), np.array([0.9, 0.8]), np.array([[0, 0.2]]), 1.0,
                             dontcare_xy=np.array([[30, 0.3]]))
    assert tp[0] and ign[1]


def test_bootstrap_units_match_reference_map():
    from src.evaluate import map_of_units
    from src.metrics import mean_ap_center
    rng = np.random.default_rng(5)
    cfg = {"matching": {"ap_thresholds_m": [0.5, 1.0, 2.0, 4.0]}}
    units, preds, gts, dcs = [], [], [], []
    for _ in range(30):
        g = rng.uniform(0, 50, (rng.integers(0, 6), 2))
        d = np.concatenate([g + rng.normal(0, 1.0, g.shape), rng.uniform(0, 50, (rng.integers(0, 4), 2))])
        s = rng.random(len(d))
        dc = rng.uniform(0, 50, (1, 2))
        preds.append((d, s)); gts.append(g); dcs.append(dc)
        per = []
        for thr in cfg["matching"]["ap_thresholds_m"]:
            tp, ign, _ = match_frame(d, s, g, thr, dc)
            per.append((s[~ign], tp[~ign], len(g)))
        units.append(per)
    ref = mean_ap_center(preds, gts, cfg["matching"]["ap_thresholds_m"], dcs)[0]
    assert map_of_units(units, cfg) == pytest.approx(ref)


def test_split_conformal_coverage_guarantee():
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(200):
        s = rng.exponential(size=301)
        q = conformal_quantile(s[:300], 0.1)
        covs.append(s[300] <= q)
    assert np.mean(covs) >= 0.86  # marginal guarantee 0.9, finite-trial slack


def test_weighted_quantile_reduces_to_unweighted():
    rng = np.random.default_rng(1)
    s = rng.normal(size=500)
    assert weighted_quantile(s, np.ones(500), 1.0, 0.1) == pytest.approx(conformal_quantile(s, 0.1), abs=1e-9)


def test_localisation_conformal_exchangeable():
    rng = np.random.default_rng(2)
    n = 4000
    r = rng.uniform(5, 60, n)
    phi = rng.normal(size=(n, 4))
    err = np.abs(rng.normal(0, 0.1 + 0.02 * r * np.exp(0.5 * phi[:, 0])))
    lc = LocalisationConformal(True).fit_scale(err[:1000], r[:1000], phi[:1000]).calibrate(err[1000:2000], r[1000:2000], phi[1000:2000], 0.1)
    res = lc.evaluate(err[2000:], r[2000:], phi[2000:])
    assert 0.87 <= res["coverage"] <= 0.93
    van = LocalisationConformal(False).fit_scale(err[:1000], r[:1000], phi[:1000]).calibrate(err[1000:2000], r[1000:2000], phi[1000:2000], 0.1)
    assert lc.evaluate(err[2000:], r[2000:], phi[2000:])["median_radius"] < van.evaluate(err[2000:], r[2000:], phi[2000:])["median_radius"] * 1.05


def test_crc_controls_risk():
    rng = np.random.default_rng(3)
    scores = rng.random((400, 5))
    loss_fn = lambda lam: (scores[:200] < lam).mean(1)
    lam = crc_threshold(loss_fn, 200, 0.2, list(np.linspace(0, 1, 101)))
    assert (scores[200:] < lam).mean() <= 0.25


def test_quantile_agrees_with_paper01():
    """PLAN A5: the two papers must share one conformal quantile. Paper 01 lives in F:/CLAUDE/AI1."""
    import importlib.util
    from pathlib import Path
    p01 = Path(__file__).resolve().parents[3] / "AI1" / "paper01-coverage-transfer" / "src" / "conformal.py"
    if not p01.exists():
        pytest.skip("Paper 01 code not present")
    spec = importlib.util.spec_from_file_location("p01_conformal", p01)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(0)
    for n in (20, 99, 500, 2000):
        s = rng.gamma(2.0, 1.0, n)
        srt = np.sort(s)
        for alpha in (0.05, 0.1, 0.2):
            k = int(np.ceil((n + 1) * (1 - alpha)))  # conformal rank (1-indexed)
            if k <= n:
                assert conformal_quantile(s, alpha) == pytest.approx(srt[k - 1])
                # Paper 01 uses np.quantile(k/n, method="higher") = the (k+1)-th order statistic: one rank
                # conservative (over-covers by <= 1/n). Documented in notes/paper01_conformal_crosscheck.md.
                assert mod.split_conformal_quantile(s, alpha) == pytest.approx(srt[min(k, n - 1)])
            else:
                assert np.isinf(conformal_quantile(s, alpha))
