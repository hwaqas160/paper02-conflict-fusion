"""E22: retrained camera 2D detector (E21) + E16 learned range/sigma/confidence heads + probes at corrected boxes (E19 step).

    python -m src.e22_combined fit | probes | candidates | router
"""
import sys

from .common import resolve


def _base(cfg):
    return resolve(cfg, "results_root") / "ts_detector"


def main():
    cmd = sys.argv[1]
    from . import e16_indomain as E16, e19_reprobe as E19
    E16.roots = lambda cfg: (_base(cfg) / "cache_cam" / "truckscenes", _base(cfg) / "cache_ch" / "truckscenes", _base(cfg) / "candidates_ch" / "truckscenes")
    E19.roots = lambda cfg: (_base(cfg) / "cache_ch" / "truckscenes", _base(cfg) / "cache_ch2" / "truckscenes", _base(cfg) / "candidates_ch2" / "truckscenes")
    if cmd == "fit":
        sys.argv = ["x", "fit"]; E16.main()
    elif cmd == "probes":
        sys.argv = ["x", "probes", "--workers", "8"]; E19.main()
    elif cmd == "candidates":
        sys.argv = ["x", "candidates"]; E19.main()
    else:
        from .e8c_router import run
        r = run(n_boot=10000, n_boot2=1000, cand_name="candidates_ch2", out_name="e22_combined_router.json", router_dir="router_ch2")
        c = r["primary_object_vs_frame_cls"]
        print({k: round(v, 3) for k, v in r["oof_map"].items()})
        print("primary", f"{c['diff']:+.4f}", [round(x, 4) for x in c["ci95"]], f"p={c['p_one_sided']:.4f}", f"folds {r['folds_object_ahead_of_frame_cls']}/5")


if __name__ == "__main__":
    main()
