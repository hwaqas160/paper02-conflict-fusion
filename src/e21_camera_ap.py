"""Camera-only Car AP (centre distance 2 m and 4 m, ROI, greedy by score) of the original COCO camera vs the retrained one."""
import json
import numpy as np
from .common import load_config, resolve


def ap_of(recs, n_gt):
    if n_gt == 0 or not recs:
        return float("nan")
    recs.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in recs]); fp = np.cumsum([1 - r[1] for r in recs])
    rec, prec = tp / n_gt, tp / np.maximum(tp + fp, 1)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def evaluate(root, thrs=(2.0, 4.0)):
    out = {}
    for tagdir in sorted(p for p in root.iterdir() if p.is_dir()):
        grp = "clear" if tagdir.name == "clear0" else "adverse"
        for f in tagdir.glob("*.npz"):
            z = np.load(f, allow_pickle=False)
            gt = z["gt_boxes"][(z["gt_cls"] == 0) & z["gt_in_roi"].astype(bool)][:, :2] if len(z["gt_boxes"]) else np.zeros((0, 2))
            k = z["camera_labels"] == 0
            xy, sc = z["camera_boxes"][k][:, :2].astype(float), z["camera_scores"][k].astype(float)
            for t in thrs:
                d = out.setdefault((grp, t), {"recs": [], "n": 0})
                d["n"] += len(gt)
                used = set()
                for i in np.argsort(-sc):
                    hit = 0
                    if len(gt):
                        dist = np.linalg.norm(gt - xy[i], axis=1)
                        for j in np.argsort(dist):
                            if dist[j] > t:
                                break
                            if j not in used:
                                used.add(j); hit = 1; break
                    d["recs"].append((sc[i], hit))
    return {f"{g}_{t:g}m": ap_of(v["recs"], v["n"]) for (g, t), v in out.items()}


cfg = load_config()
base = resolve(cfg, "results_root") / "ts_detector"
res = {"coco_raw": evaluate(base / "cache" / "truckscenes"), "coco_plus_heads_E16": evaluate(base / "cache_full2" / "truckscenes"), "retrained": evaluate(base / "cache_cam" / "truckscenes"), "combined_E22": evaluate(base / "cache_ch2" / "truckscenes")}
print(json.dumps(res, indent=2))
(resolve(cfg, "results_root") / "r4" / "e21_camera_ap.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
