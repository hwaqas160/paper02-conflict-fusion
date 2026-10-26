"""Extract every number the paper cites into paper/latex/autogen_numbers.tex as \\newcommand macros.

Run AFTER the full-resolution run: src.candidates -> src.run_experiments --exp all --retrain -> multiseed
(seeds 0-4) -> src.supplementary. No number in sections/*.tex should be typed by hand; it should be a
\\NumXxx{} macro defined here, generated straight from the JSON the experiments wrote. This is the
mechanical safeguard against a transcription mismatch between the paper and results/*.json.

    python paper/latex/make_paper_data.py
"""
import glob
import json
import numpy as np
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = Path(__file__).resolve().parent / "autogen_numbers.tex"
FIGDIR = Path(__file__).resolve().parent / "figures"

FIGURE_MAP = {
    # weather_panel is real camera/LiDAR/radar sensor data (a photograph, in the guide's sense) -> raster PNG
    # stays correct. The other five are matplotlib chart/scatter figures with no photographic content, so the
    # guide's own preference for vector artwork ("Vector drawings: EPS, PDF") applies; plots.save() already
    # writes a PDF next to every PNG, so this just points the manuscript at the vector copy already on disk.
    "figures/weather_panel.png": "weather_panel.png",
    "exp1_harm/harm_by_condition.pdf": "harm_by_condition.pdf",
    "exp2_routing/map_by_condition.pdf": "map_by_condition.pdf",
    "exp4_conformal/coverage_transfer.pdf": "coverage_transfer.pdf",
    "exp7_dropout/dropout.pdf": "dropout.pdf",
    "supplementary/forest_headline.pdf": "forest_headline.pdf",
}


def stage_figures():
    FIGDIR.mkdir(exist_ok=True)
    n = 0
    for src, dst in FIGURE_MAP.items():
        p = R / src
        if p.exists():
            shutil.copyfile(p, FIGDIR / dst)
            n += 1
        else:
            print(f"MISSING FIGURE: {p}", file=sys.stderr)
    # one two-panel figure written by the latest exp8 run (stale single-case files are deleted by exp8 itself)
    # -- a scatter plot, not a photograph, so the vector PDF goes in the manuscript, same reasoning as above.
    qp = R / "exp8_qualitative" / "qualitative_cases.pdf"
    if qp.exists():
        shutil.copyfile(qp, FIGDIR / "qual_pair.pdf")
        n += 1
    else:
        print(f"MISSING FIGURE: {qp}", file=sys.stderr)
    print(f"staged {n} figures to {FIGDIR}")


def load(*parts):
    p = R.joinpath(*parts)
    if not p.exists():
        print(f"MISSING: {p}", file=sys.stderr)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def thou(x):
    """Integer with the IEEE thousands separator for LaTeX (a thin {,})."""
    try:
        return f"{int(x):,}".replace(",", "{,}")
    except (TypeError, ValueError):
        return str(x)


def pct(x, d=2):
    return "?" if x is None else f"{100 * x:.{d}f}"


def num(x, d=3):
    return "?" if x is None else f"{x:.{d}f}"


def esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%")


TEX_LABEL = {"object": "Per-object router (ours)", "fixed_clear": "Fixed weights (status quo)",
             "frame_cls": "Per-frame weather classifier", "proben": "ProbEn \\citep{chen2022proben}",
             "dempster_shafer": "Dempster--Shafer evidential \\citep{shafer1976}",
             "clocs": "CLOCs \\citep{pang2020clocs}", "set_transformer": "Set transformer fusion \\citep{lee2019settransformer}",
             "moe_frame": "Per-frame mixture of experts \\citep{mees2016moe}", "oracle": "Oracle selector (upper bound)",
             "object_noradar": "Per-object router, radar removed"}
ABL_LABEL = {"no_absence_term": "negative-evidence (absence) term", "no_learned_precision": "learned position precisions",
             "no_harm_aux_loss": "harm auxiliary loss", "no_localisation_loss": "localization loss",
             "no_sensor_dropout": "sensor-dropout training", "no_frame_context": "frame context (DeepSets)"}


def pv(p):
    return "$<$0.001" if p is not None and p < 0.001 else num(p, 3)


def cmp_cell(v):
    """'ours minus method' cell: Holm p when we are ahead; the 95 % interval when we are behind (a p of 1.000 is opaque)."""
    if v["diff"] < 0:
        return f"{v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
    return f"{v['diff']:+.3f} ({pv(v['p_holm'])})"


def pc(p):
    """p-value carrying its own operator, for inline math: p followed by the macro gives p<0.001 or p=0.123."""
    return "<0.001" if p is not None and p < 0.001 else f"={p:.3f}"


def rows(lst):
    """LaTeX table body. A minus sign in front of a digit becomes a true math minus."""
    import re
    body = " \\\\ ".join(" & ".join(r) for r in lst) + " \\\\"
    return re.sub(r"(?<![\w\-])-(?=\d)", "$-$", body)


def r2_macros(macros):
    """Registration-2 results -> scalar macros and whole-table row macros (\\NumRtwoXRows)."""
    r2 = ("r2",)
    b = load(*r2, "r2b_baselines.json")
    if b:
        order = [a for a in ("oracle", "object", "set_transformer", "clocs", "moe_frame", "proben", "dempster_shafer",
                             "frame_cls", "fixed_clear") if a in b["map"]]
        out = []
        for a in order:
            m, v = b["map"][a], b["vs_ours"].get(a)
            out.append([TEX_LABEL[a], thou(b["params"][a]), num(m.get("clear0")), num(m.get("ADVERSE_POOLED")),
                        num(m.get("HELDOUT")), num(m.get("fog3")), num(b.get("cadc", {}).get(a)),
                        cmp_cell(v) if v else "--"])
        macros["RtwoBRows"] = rows(out)
        macros["RtwoBHeadroom"] = f"{100 * b['headroom_captured']:.0f}"
        camel = {"set_transformer": "SetTransformer", "clocs": "Clocs", "proben": "ProbEn",
                 "dempster_shafer": "DempsterShafer", "moe_frame": "MoeFrame", "frame_cls": "FrameCls",
                 "fixed_clear": "FixedClear"}
        for a, v in b["vs_ours"].items():
            suf = camel.get(a, a.title().replace("_", ""))
            macros[f"RtwoBDiff{suf}"] = num(v["diff"], 3)
            macros[f"RtwoBLo{suf}"] = num(v["ci95"][0], 3)
            macros[f"RtwoBHi{suf}"] = num(v["ci95"][1], 3)
        best = max((a for a in b["vs_ours"]), key=lambda a: b["map"][a]["ADVERSE_POOLED"])
        macros["RtwoBBestName"] = TEX_LABEL[best].split(" \\cite")[0]
        macros["RtwoBBestGap"] = num(b["vs_ours"][best]["diff"], 3)
        macros["RtwoBBestP"] = pc(b["vs_ours"][best]["p_holm"])
        macros["RtwoBNBeaten"] = str(sum(v["diff"] > 0 and v["p_holm"] < 0.05 for v in b["vs_ours"].values()))
        macros["RtwoBNCompared"] = str(len(b["vs_ours"]))
        macros["RtwoBOracleAdverse"] = num(b["map"]["oracle"]["ADVERSE_POOLED"])
    c = load(*r2, "r2c_ablation.json")
    if c:
        out = [["Full router (ours)", num(c["map"]["full"]["clear0"]), num(c["map"]["full"]["ADVERSE_POOLED"]), "--", "--"]]
        for a, v in c["vs_full"].items():
            out.append([f"w/o {ABL_LABEL.get(a, a)}", num(c["map"][a]["clear0"]), num(c["map"][a]["ADVERSE_POOLED"]),
                        f"{v['diff']:+.4f} ({pv(v['p_holm'])})", "yes" if v["contributes"] else "no"])
        macros["RtwoCRows"] = rows(out)
        macros["RtwoCNContrib"] = str(sum(v["contributes"] for v in c["vs_full"].values()))
        macros["RtwoCNTotal"] = str(len(c["vs_full"]))
        macros["RtwoCNNo"] = str(sum(not v["contributes"] for v in c["vs_full"].values()))
    e = load(*r2, "r2e_conformal.json")
    if e:
        out = []
        for g, lab in (("clear0", "clear"), ("snow3", "snow, severe"), ("fog2", "fog, moderate"), ("fog3", "fog, severe"),
                       ("SEVERE", "all severe pooled")):
            L, C = e["loc"][g], e["crc"][g]
            out.append([lab, num(L["vanilla"]["coverage"]), num(L["conflict"]["coverage"]),
                        f"{num(L['degradation_normalised']['coverage'])} / {num(L['degradation_normalised']['mean_radius'], 2)}",
                        num(C["fnr_clear_calibrated"]), num(C["fnr_floor_accept_all"]), num(C["flagged_out_of_support"]),
                        num(C["fnr_in_support"])])
        macros["RtwoERows"] = rows(out)
        f3 = e["loc"]["fog3"]
        macros["RtwoEFogCovRepaired"] = num(f3["degradation_normalised"]["coverage"])
        macros["RtwoEFogCovVanilla"] = num(f3["vanilla"]["coverage"])
        macros["RtwoEClearRadius"] = num(e["loc"]["clear0"]["degradation_normalised"]["mean_radius"], 2)
        macros["RtwoEClearRadiusVanilla"] = num(e["loc"]["clear0"]["vanilla"]["mean_radius"], 2)
        macros["RtwoEFogFloor"] = num(e["crc"]["fog3"]["fnr_floor_accept_all"])
        macros["RtwoEFogFlagged"] = f"{100 * e['crc']['fog3']['flagged_out_of_support']:.0f}"
        macros["RtwoESevereUnflaggedFNR"] = num(e["crc"]["SEVERE"]["fnr_in_support"])
        macros["RtwoEClearFlagged"] = f"{100 * e['crc']['clear0']['flagged_out_of_support']:.0f}"
        macros["RtwoEOneRefuted"] = "refuted" if e["E1_refuted"] else "not refuted"
    f = load(*r2, "r2f_cv.json")
    if f:
        out = [[str(i + 1), num(r["ADVERSE_POOLED"]["fixed_clear"]), num(r["ADVERSE_POOLED"]["frame_cls"]),
                num(r["ADVERSE_POOLED"]["object"]), f"{r['ADVERSE_POOLED']['object'] - r['ADVERSE_POOLED']['frame_cls']:+.4f}"]
               for i, r in enumerate(f["per_fold"]) if "ADVERSE_POOLED" in r]
        macros["RtwoFRows"] = rows(out)
        o = f["oof_object_vs_frame_cls"]
        macros["RtwoFDelta"], macros["RtwoFP"] = num(o["diff"], 3), pc(o["p_one_sided"])
        macros["RtwoFCILo"], macros["RtwoFCIHi"] = num(o["ci95"][0], 3), num(o["ci95"][1], 3)
        macros["RtwoFWins"] = f"{f['folds_object_beats_frame_cls']}/{f['k']}"
        macros["RtwoFOofObj"] = num(f["oof_adverse_map"]["object"])
        macros["RtwoFOofFrame"] = num(f["oof_adverse_map"]["frame_cls"])
        macros["RtwoFRefuted"] = "refuted" if f["refuted"] else "not refuted"
    g = load(*r2, "r2g_metrics.json")
    if g:
        out = [[TEX_LABEL[a], *[num(g["range_band"][a][k]) for k in ("0-20", "20-40", "40-70")], num(g["near_miss"][a])]
               for a in ("object", "frame_cls", "fixed_clear")]
        macros["RtwoGRows"] = rows(out)
        macros["RtwoGLatMedian"] = num(g["latency_ms"]["object"]["median"], 1)
        macros["RtwoGLatPninetyfive"] = num(g["latency_ms"]["object"]["p95"], 1)
        macros["RtwoGNearMissObj"] = pct(g["near_miss"]["object"], 1)
        macros["RtwoGNearMissFixed"] = pct(g["near_miss"]["fixed_clear"], 1)
        macros["RtwoGFarObj"] = num(g["range_band"]["object"]["40-70"])
        macros["RtwoGFarFixed"] = num(g["range_band"]["fixed_clear"]["40-70"])
    a = load("truckscenes", "r2", "r2a_truckscenes.json")
    if a:
        om = a["oof_map"]
        pools = ["clear0", "ADVERSE_POOLED"]
        order = [x for x in ("oracle", "object", "object_noradar", "set_transformer", "clocs", "moe_frame", "proben",
                             "dempster_shafer", "frame_cls", "fixed_clear") if x in om["ADVERSE_POOLED"]]
        out = []
        for x in order:
            if x == "frame_cls":
                d = f"{a['A1']['diff']:+.3f} ({pv(a['A1']['p_one_sided'])})"
            elif x == "object_noradar":
                d = f"{a['A3']['diff']:+.3f} ({pv(a['A3']['p_one_sided'])})"
            elif x in a.get("B_vs_ours", {}):
                d = cmp_cell(a["B_vs_ours"][x])
            else:
                d = "--"
            out.append([TEX_LABEL[x]] + [num(om.get(p, {}).get(x)) for p in pools] +
                       [num(a["official"]["map"]["ADVERSE_POOLED"].get(x)), d])
        macros["RtwoARows"] = rows(out)
        macros["RtwoANFrames"] = f"{a['n_frames']:,}".replace(",", "{,}")
        macros["RtwoANScenes"] = str(a["n_scenes"])
        macros["RtwoAScenesCond"] = ", ".join(f"{k.rstrip('01')} {v}" for k, v in a["scenes_per_condition"].items())
        for key, lab in (("A1", "One"), ("A3", "Three")):
            macros[f"RtwoA{lab}Delta"] = num(a[key]["diff"], 3)
            macros[f"RtwoA{lab}P"] = pc(a[key]["p_one_sided"])
            macros[f"RtwoA{lab}CILo"], macros[f"RtwoA{lab}CIHi"] = num(a[key]["ci95"][0], 3), num(a[key]["ci95"][1], 3)
            macros[f"RtwoA{lab}Refuted"] = "refuted" if a[key]["refuted"] else "not refuted"
        macros["RtwoAOneNScenes"] = str(a["A1"]["n_scenes"])
        macros["RtwoAOneFolds"] = f"{sum(x > 0 for x in a['fold_deltas_A1'])}/{len(a['fold_deltas_A1'])}"
        macros["RtwoATwoHarm"] = pct(a["A2"]["harm_rate"])
        macros["RtwoATwoHelp"] = pct(a["A2"]["help_rate"])
        macros["RtwoATwoPairs"] = f"{a['A2']['n_pairs']:,}".replace(",", "{,}")
        macros["RtwoATwoLidarHarm"] = pct(a["A2"]["per_modality"]["lidar"]["harm_rate"])
        macros["RtwoATwoLidarHelp"] = pct(a["A2"]["per_modality"]["lidar"]["help_rate"])
        macros["RtwoATwoRadarHarm"] = pct(a["A2"]["per_modality"]["radar"]["harm_rate"])
        macros["RtwoATwoCameraHarm"] = pct(a["A2"]["per_modality"]["camera"]["harm_rate"])
        if "set_transformer" in a.get("B_vs_ours", {}):
            st = a["B_vs_ours"]["set_transformer"]
            macros["RtwoAStDiff"] = f"{st['diff']:+.3f}"
            macros["RtwoAStLo"], macros["RtwoAStHi"] = f"{st['ci95'][0]:+.3f}", f"{st['ci95'][1]:+.3f}"
            macros["RtwoAStAdverse"] = num(om["ADVERSE_POOLED"]["set_transformer"])
            macros["RtwoAClocsAdverse"] = num(om["ADVERSE_POOLED"]["clocs"])
            macros["RtwoAOracleAdverse"] = num(om["ADVERSE_POOLED"]["oracle"])
        macros["RtwoANoRadarAdverse"] = num(om["ADVERSE_POOLED"]["object_noradar"])
        macros["RtwoACondScenes"] = ", ".join(f"{k.rstrip('01').replace('night', 'night')} {v}" for k, v in a["scenes_per_condition"].items())
        if "E" in a:
            sn, fg = a["E"]["snow1"], a["E"]["fog1"]
            macros["RtwoAEsnowVanilla"], macros["RtwoAEsnowRepaired"] = num(sn["cov_vanilla"]), num(sn["cov_degradation_normalised"])
            macros["RtwoAEsnowFlagged"] = f"{100 * sn['flagged_out_of_support']:.0f}"
            macros["RtwoAEsnowFnr"], macros["RtwoAEsnowFnrUnflagged"] = num(sn["fnr_clear_calibrated"]), num(sn["fnr_in_support"])
            macros["RtwoAEclearFnr"] = num(a["E"]["clear0"]["fnr_clear_calibrated"])
            macros["RtwoAErainRepaired"] = num(a["E"]["rain1"]["cov_degradation_normalised"])
        macros["RtwoATwoCILo"], macros["RtwoATwoCIHi"] = pct(a["A2"]["harm_ci"][0]), pct(a["A2"]["harm_ci"][1])
        macros["RtwoATwoRefuted"] = "refuted" if a["A2"]["refuted"] else "not refuted"
        macros["RtwoAObjAdverse"] = num(om["ADVERSE_POOLED"]["object"])
        macros["RtwoAFrameAdverse"] = num(om["ADVERSE_POOLED"]["frame_cls"])
        macros["RtwoAFixedAdverse"] = num(om["ADVERSE_POOLED"]["fixed_clear"])
        if "headroom_captured" in a:
            macros["RtwoAHeadroom"] = f"{100 * a['headroom_captured']:.0f}"
            macros["RtwoANBeaten"] = str(sum(v["diff"] > 0 and v["p_holm"] < 0.05 for v in a["B_vs_ours"].values()))
            macros["RtwoANCompared"] = str(len(a["B_vs_ours"]))
        if "C_vs_full" in a:
            macros["RtwoATsCRows"] = rows([[f"w/o {ABL_LABEL.get(n, n)}", num(a["C_map"][n]),
                                            f"{v['diff']:+.4f} ({pv(v['p_holm'])})", "yes" if v["contributes"] else "no"]
                                           for n, v in a["C_vs_full"].items()])
            macros["RtwoATsCNContrib"] = str(sum(v["contributes"] for v in a["C_vs_full"].values()))
            if c:  # one table, both datasets: Boreas (clear, adverse, diff, contributes) then TruckScenes (adverse, diff, contributes)
                both = [["Full router (ours)", num(c["map"]["full"]["clear0"]), num(c["map"]["full"]["ADVERSE_POOLED"]), "--", "--",
                         num(a["oof_map"]["ADVERSE_POOLED"]["object"]), "--", "--"]]
                for n, v in c["vs_full"].items():
                    t = a["C_vs_full"][n]
                    both.append([f"w/o {ABL_LABEL.get(n, n)}", num(c["map"][n]["clear0"]), num(c["map"][n]["ADVERSE_POOLED"]),
                                 f"{v['diff']:+.4f} ({pv(v['p_holm'])})", "yes" if v["contributes"] else "no",
                                 num(a["C_map"][n]), f"{t['diff']:+.4f} ({pv(t['p_holm'])})", "yes" if t["contributes"] else "no"])
                macros["RtwoCBothRows"] = rows(both)
        if "E" in a:
            macros["RtwoATsERows"] = rows([[t.rstrip("01"), num(v["cov_vanilla"]), num(v["cov_conflict"]),
                                            f"{num(v['cov_degradation_normalised'])} / {num(v['rad_degradation_normalised'], 2)}",
                                            num(v["fnr_clear_calibrated"]), num(v["fnr_floor_accept_all"]),
                                            num(v["flagged_out_of_support"]), num(v["fnr_in_support"])]
                                           for t, v in a["E"].items()])
        o = a["official"]
        macros["RtwoAOffDelta"], macros["RtwoAOffP"] = num(o["A1"]["diff"], 3), pc(o["A1"]["p_one_sided"])
        macros["RtwoAOffObj"] = num(o["map"]["ADVERSE_POOLED"]["object"])
        macros["RtwoAOffFrame"] = num(o["map"]["ADVERSE_POOLED"]["frame_cls"])
    if b and a:
        # R2-B1 rule: refuted for a baseline if that baseline beats ours (whole 95 % CI of ours-minus-baseline below 0)
        short = {"set_transformer": "the set-transformer", "clocs": "CLOCs", "proben": "ProbEn",
                 "dempster_shafer": "Dempster--Shafer", "moe_frame": "the mixture of experts",
                 "frame_cls": "the per-frame classifier", "fixed_clear": "the status quo"}
        beaten_by = [f"{short.get(k, k)} on {ds}" for ds, src in (("Boreas", b["vs_ours"]), ("TruckScenes", a.get("B_vs_ours", {})))
                     for k, v in src.items() if v["ci95"][1] < 0]
        macros["RtwoBOneVerdict"] = "not refuted" if not beaten_by else "refuted for " + " and ".join(beaten_by)
    d = load(*r2, "r2d_cadc.json")
    if d:
        arms = ("fixed_clear", "frame_cls", "object")
        out = [["zero-shot, raw probes"] + [num(d["D1_raw"]["map"][x]) for x in arms] +
               [f"{d['D1_raw']['object_vs_frame_cls']['diff']:+.4f} ({pv(d['D1_raw']['object_vs_frame_cls']['p_one_sided'])})"],
               ["zero-shot, resolution-normalized"] + [num(d["D1_normalised"]["map"][x]) for x in arms] +
               [f"{d['D1_normalised']['object_vs_frame_cls']['diff']:+.4f} ({pv(d['D1_normalised']['object_vs_frame_cls']['p_one_sided'])})"]]
        for k, v in d["D2"].items():
            o = v["object_vs_frame_cls"]
            out.append([f"few-shot, $k={k}$ drives"] + [num(v["pooled_map"][x]) for x in arms] +
                       [f"{o['diff']:+.4f} ({pv(o['p_one_sided'])})"])
        macros["RtwoDRows"] = rows(out)
        o = d["D1_normalised"]["object_vs_frame_cls"]
        macros["RtwoDOneDelta"], macros["RtwoDOneP"] = num(o["diff"], 3), pc(o["p_one_sided"])
        macros["RtwoDOneRefuted"] = "refuted" if d["D1_refuted"] else "not refuted"
        kmax = str(max(int(k) for k in d["D2"]))
        o = d["D2"][kmax]["object_vs_frame_cls"]
        macros["RtwoDTwoDelta"], macros["RtwoDTwoP"] = num(o["diff"], 3), pc(o["p_one_sided"])
        macros["RtwoDTwoRefuted"] = "refuted" if d["D2_refuted"] else "not refuted"
        macros["RtwoDTwoMonotone"] = "yes" if d["D2_monotone"] else "no"
        macros["RtwoDNDrives"] = str(d["n_drives"])


R3_LABEL = {"original": "Original router", "V0": "Original router, re-fit", "V1": "V1: attention context",
            "V2": "V2: bounded correction (final)", "V3": "V3: V1 + V2", "set_transformer": "Set transformer (black box)",
            "set_noattn": "Set transformer, attention removed", "variant_rule_only": "V2, correction switched off"}


def r3_macros(macros):
    """Registration-3 results (closing the gap to the set-transformer)."""
    d = load("truckscenes_dev", "r3", "r3_dev.json")
    if d:
        order = ["original", "V0", "V1", "V2", "V3", "set_transformer", "set_noattn"]
        macros["RthreeDevRows"] = rows([[R3_LABEL[a], f"{d['params'][a]:,}".replace(",", "{,}"), num(d["map"]["ALL"][a]),
                                         num(d["map"]["ADVERSE"][a]), num(d["map"]["clear0"][a])] for a in order])
        macros["RthreeDevScenes"] = str(d["n_dev_test_scenes"])
        macros["RthreeDevSelected"] = str(d["selected"])
        macros["RthreeDevNoAttnAll"] = num(d["map"]["ALL"]["set_noattn"])
        macros["RthreeDevStAll"] = num(d["map"]["ALL"]["set_transformer"])
        macros["RthreeDevVOneAll"] = num(d["map"]["ALL"]["V1"])
        macros["RthreeDevVZeroAll"] = num(d["map"]["ALL"]["V0"])
        macros["RthreeDevVTwoAll"] = num(d["map"]["ALL"]["V2"])
        macros["RthreeVOneParams"] = f"{d['params']['V1']:,}".replace(",", "{,}")
        macros["RthreeVTwoParams"] = f"{d['params']['V2']:,}".replace(",", "{,}")
        macros["RthreeStParams"] = f"{d['params']['set_transformer']:,}".replace(",", "{,}")
    t = load("truckscenes", "r3", "r3_final_truckscenes.json")
    if t:
        order = ["original", "set_transformer", "V2", "variant_rule_only"]
        vs = t["vs"]
        def dcell(a):
            if a == "V2":
                return "--"
            v = vs[a]
            return f"{v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
        macros["RthreeTsRows"] = rows([[R3_LABEL[a], f"{t['params'][a]:,}".replace(",", "{,}"), num(t["map"]["clear0"][a]),
                                        num(t["map"]["ADVERSE"][a]), num(t["map"]["ALL"][a]), dcell(a)] for a in order])
        macros["RthreeTsScenes"] = str(t["n_test_scenes"])
        macros["RthreeTsAdverseScenes"] = str(t["n_adverse_scenes"])
        macros["RthreeTsVTwo"] = num(t["map"]["ADVERSE"]["V2"])
        macros["RthreeTsSt"] = num(t["map"]["ADVERSE"]["set_transformer"])
        macros["RthreeTsOrig"] = num(t["map"]["ADVERSE"]["original"])
        macros["RthreeTsRuleOnly"] = num(t["map"]["ADVERSE"]["variant_rule_only"])
        macros["RthreeTsVsStDiff"] = f"{vs['set_transformer']['diff']:+.3f}"
        macros["RthreeTsVsStLo"], macros["RthreeTsVsStHi"] = f"{vs['set_transformer']['ci95'][0]:+.3f}", f"{vs['set_transformer']['ci95'][1]:+.3f}"
        macros["RthreeTsVsOrigDiff"] = f"{vs['original']['diff']:+.3f}"
        macros["RthreeTsVsOrigP"] = pc(vs["original"]["p_one_sided"])
        macros["RthreeNonInfLo"] = f"{t['R3_1_lower_bound']:+.3f}"
        macros["RthreeNonInfRefuted"] = "refuted" if t["R3_1_refuted"] else "not refuted"
        macros["RthreeCorrLogit"] = num(t["correction"]["mean_abs_logit_correction"], 2)
        macros["RthreeCorrMeters"] = num(t["correction"]["mean_position_correction_m"], 2)
        macros["RthreeLatVTwo"] = num(t["latency_ms"]["variant"]["median"], 1)
        macros["RthreeLatOrig"] = num(t["latency_ms"]["original"]["median"], 1)
    b = load("r3", "r3_final_boreas.json")
    if b:
        order = ["original", "set_transformer", "V2", "variant_rule_only"]
        macros["RthreeBoRows"] = rows([[R3_LABEL[a], f"{b['params'][a]:,}".replace(",", "{,}"), num(b["map"]["clear0"][a]),
                                        num(b["map"]["ADVERSE_POOLED"][a]), num(b["map"]["HELDOUT"][a]),
                                        num(b["map"]["fog3"][a])] for a in order])
        macros["RthreeBoVTwo"] = num(b["map"]["ADVERSE_POOLED"]["V2"])
        macros["RthreeBoOrig"] = num(b["map"]["ADVERSE_POOLED"]["original"])
        macros["RthreeBoSt"] = num(b["map"]["ADVERSE_POOLED"]["set_transformer"])
        macros["RthreeBoVsOrigDiff"] = f"{b['vs']['original']['diff']:+.3f}"
        macros["RthreeBoVsOrigLo"] = num(b["vs"]["original"]["ci95"][0], 3)
        macros["RthreeBoVsOrigHi"] = num(b["vs"]["original"]["ci95"][1], 3)
        macros["RthreeBoVsOrigP"] = pc(b["vs"]["original"]["p_one_sided"])
        macros["RthreeBoVsStDiff"] = f"{b['vs']['set_transformer']['diff']:+.3f}"
        macros["RthreeBoVsStP"] = pc(b["vs"]["set_transformer"]["p_one_sided"])
        macros["RthreeBoRefuted"] = "refuted" if b["R3_2_refuted"] else "not refuted"
    macros["RthreeCvNote"] = ""
    c = load("truckscenes", "r3", "r3_final_cv.json")
    if c:
        macros["RthreeCvVTwo"] = num(c["oof_map"]["V2"])
        macros["RthreeCvSt"] = num(c["oof_map"]["set_transformer"])
        macros["RthreeCvOrig"] = num(c["oof_map"]["original"])
        macros["RthreeCvVsStDiff"] = f"{c['vs']['set_transformer']['diff']:+.3f}"
        macros["RthreeCvVsStLo"], macros["RthreeCvVsStHi"] = f"{c['vs']['set_transformer']['ci95'][0]:+.3f}", f"{c['vs']['set_transformer']['ci95'][1]:+.3f}"
        macros["RthreeCvVsOrigDiff"] = f"{c['vs']['original']['diff']:+.3f}"
        macros["RthreeCvWins"] = f"{c['folds_variant_ge_set_transformer']}/{len(c['per_fold'])}"
        macros["RthreeCvNote"] = (
            "Under five-fold cross-validation over all 598 scenes -- a robustness view that is \\emph{not} independent of "
            "development, because its folds contain development scenes -- V2 reaches " + macros["RthreeCvVTwo"] +
            " against " + macros["RthreeCvSt"] + " for the set-transformer and " + macros["RthreeCvOrig"] +
            " for the original router, at least matching the set-transformer in " + macros["RthreeCvWins"] +
            " folds (difference " + macros["RthreeCvVsStDiff"] + ", 95\\,\\% CI [" + macros["RthreeCvVsStLo"] + ", " +
            macros["RthreeCvVsStHi"] + "]).")


def r4_macros(macros):
    a = load("r4", "r4_1_decision_unit.json")
    if a:
        macros["RfourOneObjMap"] = num(a["map_object"])
        macros["RfourOnePoolMap"] = num(a["map_frame_pooled"])
        macros["RfourOneDiff"] = f"{a['primary_scene_cluster']['diff']:+.3f}"
        macros["RfourOneLo"] = f"{a['primary_scene_cluster']['ci95'][0]:+.3f}"
        macros["RfourOneHi"] = f"{a['primary_scene_cluster']['ci95'][1]:+.3f}"
        macros["RfourOneP"] = pc(a["primary_scene_cluster"]["p_one_sided"])
        macros["RfourOneNScenes"] = str(a["primary_scene_cluster"]["n_scenes"])
        macros["RfourOnePoolParams"] = f"{a['params']['frame_pooled']:,}".replace(",", "{,}")
        macros["RfourOneRefuted"] = "refuted" if a["refuted"] else "not refuted"
    b = load("truckscenes", "r4", "r4_2_truckscenes_decision_unit.json")
    if b:
        macros["RfourTwoObjMap"] = num(b["oof_map_object"])
        macros["RfourTwoPoolMap"] = num(b["oof_map_frame_pooled"])
        macros["RfourTwoDiff"] = f"{b['primary_scene_cluster']['diff']:+.3f}"
        macros["RfourTwoLo"] = f"{b['primary_scene_cluster']['ci95'][0]:+.3f}"
        macros["RfourTwoHi"] = f"{b['primary_scene_cluster']['ci95'][1]:+.3f}"
        macros["RfourTwoP"] = pc(b["primary_scene_cluster"]["p_one_sided"])
        macros["RfourTwoNScenes"] = str(b["primary_scene_cluster"]["n_scenes"])
        macros["RfourTwoWins"] = f"{b['folds_object_ahead']}/{len(b['per_fold'])}"
        macros["RfourTwoRefuted"] = "refuted" if b["refuted"] else "not refuted"
    camsum = load("truckscenes", "r4", "candidates_stronger_cam", "truckscenes", "_summary.json")
    if camsum:
        n_box = sum(v["n_camera_boxes"] for v in camsum.values())
        n_over = sum(v["n_lidar_overridden"] for v in camsum.values())
        macros["RfourFourOverrideRate"] = pct(n_over / n_box) if n_box else "?"
    d = load("r4", "stronger_cam", "r4", "r4_4_stronger_camera.json")
    if d:
        macros["RfourFourObjMap"] = num(d["oof_map_object"])
        macros["RfourFourFrameMap"] = num(d["oof_map_frame_cls"])
        macros["RfourFourDiff"] = f"{d['primary_scene_cluster']['diff']:+.3f}"
        macros["RfourFourLo"] = f"{d['primary_scene_cluster']['ci95'][0]:+.3f}"
        macros["RfourFourHi"] = f"{d['primary_scene_cluster']['ci95'][1]:+.3f}"
        macros["RfourFourP"] = pc(d["primary_scene_cluster"]["p_one_sided"])
        macros["RfourFourNScenes"] = str(d["primary_scene_cluster"]["n_scenes"])
        macros["RfourFourWins"] = f"{d['folds_object_ahead']}/{len(d['per_fold'])}"
        macros["RfourFourRefuted"] = "refuted" if d["refuted"] else "not refuted"
    e3 = load("r4", "e3_per_scene_boreas.json")
    if e3:
        diffs = [r["diff"] for r in e3["rows"]]
        macros["EThreeNScenes"] = str(e3["n_scenes"])
        macros["EThreeNAhead"] = str(e3["n_object_ahead"])
        macros["EThreeMinDiff"] = num(min(diffs), 3)
        macros["EThreeMaxDiff"] = num(max(diffs), 3)
    c = load("r4", "r4_3_support_check.json")
    if c:
        au = c["aurocs"]
        macros["RfourThreeSupportAuroc"] = num(au["support_check"])
        macros["RfourThreeMahaAuroc"] = num(au["mahalanobis"])
        macros["RfourThreeKnnAuroc"] = num(au["knn10"])
        macros["RfourThreeLogitAuroc"] = num(au["logistic_density_ratio"])
        macros["RfourThreeBestBaseline"] = num(c["best_baseline"])
        macros["RfourThreeNCal"] = str(c["n_cal"])
        macros["RfourThreeNPos"] = str(c["n_pos"])
        macros["RfourThreeNNeg"] = str(c["n_neg"])
        macros["RfourThreeRefuted"] = "refuted" if c["refuted"] else "not refuted"


E10_LABEL = {"frame_cls": "Per-frame weather classifier",
             "geom": "Per-object: range and azimuth only",
             "geom_conf": "Per-object: + detector scores and presence",
             "geom_clocs": "Per-object: + detector and conflict features, no probes",
             "object": "\\textbf{Per-object: full router (ours)}"}


def e10_macros(macros):
    bo = load("r4", "e10_minimal_boreas.json")
    ts = load("truckscenes", "r4", "e10_minimal_truckscenes.json")
    order = ("frame_cls", "geom", "geom_conf", "geom_clocs", "object")

    def cell(res, arm, key):
        if res is None:
            return "--"
        if arm == "frame_cls":
            return "--"
        v = res["object_vs_frame_cls"] if arm == "object" else res["vs_frame_cls"][arm]
        return f"{v['diff']:+.3f} ({v['scenes_ahead']}/{v['n_scenes']})"

    table = []
    for arm in order:
        row = [E10_LABEL[arm], f"{bo['params'][arm]:,}".replace(",", "{,}") if bo else "--",
               num(bo["map"][arm]) if bo else "--", cell(bo, arm, "bo"),
               num(ts["oof_map"][arm]) if ts else "--", cell(ts, arm, "ts")]
        table.append(row)
    macros["EtenRows"] = rows(table)
    short = {"frame_cls": "Per-frame classifier", "geom": "Range, azimuth", "geom_conf": "+ detector scores",
             "geom_clocs": "+ conflict features", "object": "\\textbf{Full router}"}
    macros["EtenRowsShort"] = rows([[short[a]] + r[1:] for a, r in zip(order, table)])
    for tag, res in (("Bo", bo), ("Ts", ts)):
        if not res:
            continue
        full = res["object_vs_frame_cls"]
        clocs = res["vs_frame_cls"]["geom_clocs"]
        macros[f"Eten{tag}FullDiff"] = f"{full['diff']:+.3f}"
        macros[f"Eten{tag}NScenes"] = str(full["n_scenes"])
        macros[f"Eten{tag}FullAhead"] = str(full["scenes_ahead"])
        macros[f"Eten{tag}FullSignP"] = pc(full["sign_test_p_two_sided"])
        macros[f"Eten{tag}GeomDiff"] = f"{res['vs_frame_cls']['geom']['diff']:+.3f}"
        macros[f"Eten{tag}ConfDiff"] = f"{res['vs_frame_cls']['geom_conf']['diff']:+.3f}"
        macros[f"Eten{tag}ClocsDiff"] = f"{clocs['diff']:+.3f}"
        macros[f"Eten{tag}ClocsAhead"] = str(clocs["scenes_ahead"])
        macros[f"Eten{tag}ClocsSignP"] = pc(clocs["sign_test_p_two_sided"])
        macros[f"Eten{tag}ClocsShare"] = pct(clocs["diff"] / full["diff"], 0)
        macros[f"Eten{tag}GeomShare"] = pct(res["vs_frame_cls"]["geom"]["diff"] / full["diff"], 0)
        inc = res["object_vs_minimal"]["geom_clocs"]
        macros[f"Eten{tag}IncDiff"] = f"{inc['diff']:+.3f}"
        macros[f"Eten{tag}IncLo"] = f"{inc['ci95'][0]:+.3f}"
        macros[f"Eten{tag}IncHi"] = f"{inc['ci95'][1]:+.3f}"
        macros[f"Eten{tag}IncAhead"] = str(inc["scenes_ahead"])
        macros[f"Eten{tag}IncBehind"] = str(inc["scenes_behind"])
        macros[f"Eten{tag}IncBootP"] = pc(inc["p_one_sided"])
        macros[f"Eten{tag}IncSignP"] = pc(inc["sign_test_p_two_sided"])
        macros[f"Eten{tag}NBoot"] = f"{res['n_boot']:,}".replace(",", "{,}")


def e11_macros(macros):
    r = load("truckscenes", "r4", "e11_per_condition.json")
    if not r:
        return
    names = {"fog1": "Fog", "snow1": "Snow", "rain1": "Rain", "night1": "Night", "clear0": "Clear (reference)"}
    table = []
    for t in ("fog1", "snow1", "rain1", "night1", "clear0"):
        c = r["conditions"].get(t)
        if not c:
            continue
        table.append([names[t], str(c["n_scenes"]), num(c["map_object"]), num(c["map_frame_cls"]),
                      f"{c['diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]",
                      f"{c['scenes_ahead']}/{c['n_scenes']}", pv(c["sign_test_p_two_sided"])])
    macros["EelevenRows"] = rows(table)
    macros["EelevenRowsShort"] = rows([[r[0].replace(" (reference)", ""), r[1], r[2], r[3], r[4], r[5]] for r in table])
    macros["EelevenNBoot"] = f"{r['n_boot']:,}".replace(",", "{,}")
    adv = [r["conditions"][t] for t in ("fog1", "snow1", "rain1", "night1") if t in r["conditions"]]
    macros["EelevenMinDiff"] = f"{min(c['diff'] for c in adv):+.3f}"
    macros["EelevenMaxDiff"] = f"{max(c['diff'] for c in adv):+.3f}"
    macros["EelevenFogAhead"] = f"{r['conditions']['fog1']['scenes_ahead']}/{r['conditions']['fog1']['n_scenes']}"
    macros["EelevenSnowAhead"] = f"{r['conditions']['snow1']['scenes_ahead']}/{r['conditions']['snow1']['n_scenes']}"


def e12_macros(macros):
    r = load("truckscenes", "r4", "e12_matched_radius.json")
    if not r:
        return
    for tag, key in (("Snow", "snow1"), ("Fog", "fog1"), ("Rain", "rain1")):
        c = r["conditions"].get(key)
        if not c:
            continue
        macros[f"Etwelve{tag}NormCov"] = num(c["coverage_normalised"])
        macros[f"Etwelve{tag}NormRad"] = num(c["mean_radius_normalised"], 2)
        macros[f"Etwelve{tag}VanRad"] = num(c["mean_radius_vanilla"], 2)
        macros[f"Etwelve{tag}VanScale"] = num(c["scale_to_match_vanilla"], 2)
        macros[f"Etwelve{tag}VanMatched"] = num(c["coverage_vanilla_at_matched_radius"])
        macros[f"Etwelve{tag}ConfMatched"] = num(c["coverage_conflict_at_matched_radius"])


def e17_macros(macros):
    """E17 lean router and E18 CADC-native router."""
    r = load("r4", "e17_lean_both.json")
    if r:
        b, s = r["boreas"], r["truckscenes"]
        macros["SeventeenParams"] = thou(b["params"]["lean"])
        macros["SeventeenBoreasMap"] = num(b["map"]["lean"])
        macros["SeventeenBoreasVsFull"] = f"{b['lean_vs_full']['diff']:+.4f}"
        macros["SeventeenTsMap"] = num(s["oof_map"]["lean"])
        macros["SeventeenTsFullMap"] = num(s["oof_map"]["full"])
        macros["SeventeenTsVsFull"] = f"{s['lean_vs_full']['diff']:+.4f}"
        macros["SeventeenTsVsFrame"] = f"{s['lean_vs_frame_cls']['diff']:+.4f}"
    c = load("r4", "e18_cadc_native.json")
    if c:
        o = c["object_vs_fixed"]
        macros["EighteenDiff"] = f"{o['diff']:+.4f}"
        macros["EighteenLo"] = f"{o['ci95'][0]:+.4f}"
        macros["EighteenHi"] = f"{o['ci95'][1]:+.4f}"
        macros["EighteenFolds"] = str(c["folds_object_ahead"])


def e16_macros(macros):
    """E16: fully in-domain suite (cross-fitted LiDAR plus learned camera and radar heads)."""
    r = load("ts_detector", "r4", "e16_fully_indomain_router.json")
    if r:
        c = r["primary_object_vs_frame_cls"]
        macros["SixteenObj"] = num(r["oof_map"]["object"])
        macros["SixteenFrame"] = num(r["oof_map"]["frame_cls"])
        macros["SixteenDiff"] = f"{c['diff']:+.4f}"
        macros["SixteenLo"] = f"{c['ci95'][0]:+.4f}"
        macros["SixteenHi"] = f"{c['ci95'][1]:+.4f}"
        macros["SixteenP"] = pc(c["p_one_sided"])
        macros["SixteenFolds"] = str(r["folds_object_ahead_of_frame_cls"])
        macros["SixteenPooledDiff"] = f"{r['object_vs_frame_pooled']['diff']:+.4f}"
    r9 = load("ts_detector", "r4", "e19_reprobe_router.json")
    if r9:
        c = r9["primary_object_vs_frame_cls"]
        macros["SixteenObj"] = num(r9["oof_map"]["object"])
        macros["SixteenFrame"] = num(r9["oof_map"]["frame_cls"])
        macros["SixteenDiff"] = f"{c['diff']:+.4f}"
        macros["SixteenLo"] = f"{c['ci95'][0]:+.4f}"
        macros["SixteenHi"] = f"{c['ci95'][1]:+.4f}"
        macros["SixteenP"] = pc(c["p_one_sided"])
        macros["SixteenFolds"] = str(r9["folds_object_ahead_of_frame_cls"])
    q20 = load("truckscenes", "r4", "e20_quantile.json")
    if q20:
        macros["TwentyObj"] = num(q20["map"]["object"])
        macros["TwentyFrame"] = num(q20["map"]["frame_cls"])
    own = load("truckscenes", "r4", "e15b_selective_ts_degradation-own.json")
    if own:
        macros["EfifteenOwnSnowFnr"] = num(own["conditions"]["snow1"]["fnr_accepted"])
    r22 = load("ts_detector", "r4", "e22_combined_router.json")
    if r22:
        c = r22["primary_object_vs_frame_cls"]
        macros["TwentytwoDiff"] = f"{c['diff']:+.4f}"
        macros["TwentytwoLo"] = f"{c['ci95'][0]:+.4f}"
        macros["TwentytwoHi"] = f"{c['ci95'][1]:+.4f}"
        macros["TwentytwoFolds"] = str(r22["folds_object_ahead_of_frame_cls"])
    ap = load("r4", "e21_camera_ap.json")
    if ap:
        macros["CamApRaw"] = num(ap["coco_raw"]["adverse_2m"], 2)
        macros["CamApComb"] = num(ap["combined_E22"]["adverse_2m"], 2)
    q = load("r4", "e16_head_quality.json")
    if q:
        macros["SixteenCamAurocOld"] = num(q["camera_matched_auroc"]["original"], 2)
        macros["SixteenCamAurocNew"] = num(q["camera_matched_auroc"]["learned"], 2)
        macros["SixteenRadAurocOld"] = num(q["radar_auroc"]["original"], 2)
        macros["SixteenRadAurocNew"] = num(q["radar_auroc"]["learned"], 2)


def e15_macros(macros):
    """E15 / E15b: selective conformal risk control (abstain on a label-free score)."""
    z = load("r4", "e15_selective.json")
    if z:
        c = z["conditions"]
        macros["EfifteenAlpha"] = f"{z['alpha']:.2f}"
        for key, tag in (("Clear", "clear0"), ("FogTwo", "fog2"), ("RainThree", "rain3"), ("SnowThree", "snow3")):
            macros[f"Efifteen{key}Fnr"] = num(c[tag]["fnr_accepted"])
        macros["EfifteenFogThreeAccepted"] = pct(c["fog3"]["accepted"], 0)
        macros["EfifteenFogThreeFnrAll"] = num(c["fog3"]["fnr_all_frames"])
    b = load("r2", "r2b_baselines.json")
    if b and "cadc" in b:
        macros["CadcOracleMap"] = num(b["cadc"]["oracle"])
        macros["CadcOracleHeadroom"] = num(b["cadc"]["oracle"] - b["cadc"]["fixed_clear"])
    r = load("truckscenes", "r4", "e15b_selective_ts.json")
    if r:
        c = r["conditions"]
        for key, tag in (("Fog", "fog1"), ("Night", "night1"), ("Rain", "rain1"), ("Snow", "snow1")):
            macros[f"EfifteenB{key}Fnr"] = num(c[tag]["fnr_accepted"])
        macros["EfifteenBSnowAccepted"] = pct(c["snow1"]["accepted"], 0)


def e13_macros(macros):
    z = load("truckscenes", "r4", "e13_sim2real.json")
    if z:
        macros["EthirteenObj"] = num(z["map"]["object"])
        macros["EthirteenFrame"] = num(z["map"]["frame_cls"])
        macros["EthirteenFixed"] = num(z["map"]["fixed_clear"])
        c = z["object_vs_frame_cls"]
        macros["EthirteenDiff"] = f"{c['diff']:+.3f}"
        macros["EthirteenBehind"] = str(c["scenes_behind"])
        macros["EthirteenNScenes"] = str(c["n_scenes"])
        macros["EthirteenGeomDiff"] = f"{z['minimal_vs_frame_cls']['geom']['diff']:+.3f}"
        macros["EthirteenClocsDiff"] = f"{z['minimal_vs_frame_cls']['geom_clocs']['diff']:+.3f}"
    r = load("truckscenes", "r4", "e13_sim2real_recentred.json")
    if r:
        macros["EthirteenRecObj"] = num(r["map"]["object"])
        macros["EthirteenRecFrame"] = num(r["map"]["frame_cls"])
        c = r["object_vs_frame_cls"]
        macros["EthirteenRecDiff"] = f"{c['diff']:+.3f}"
        macros["EthirteenRecLo"] = f"{c['ci95'][0]:+.3f}"
        macros["EthirteenRecHi"] = f"{c['ci95'][1]:+.3f}"
        macros["EthirteenRecAhead"] = str(c["scenes_ahead"])
        macros["EthirteenRecBehind"] = str(c["scenes_behind"])
        macros["EthirteenRecSignP"] = pc(c["sign_test_p_two_sided"])
    n = load("truckscenes", "r4", "e13_sim2real_neutral.json")
    if n:
        macros["EthirteenNeuObj"] = num(n["map"]["object"])


def e8c_macros(macros):
    ev = [load("ts_detector", f"eval_fold{k}.json") for k in (0, 1)]
    if all(ev):
        macros["EeightcKittiAp"] = num(np.mean([e["kitti"]["mean"] for e in ev]))
        macros["EeightcIndomainAp"] = num(np.mean([e["in_domain"]["mean"] for e in ev]))
        macros["EeightcKittiApFive"] = num(np.mean([e["kitti"]["0.5"] for e in ev]))
        macros["EeightcIndomainApFive"] = num(np.mean([e["in_domain"]["0.5"] for e in ev]))
        for cond, nm in (("snow", "Snow"), ("fog", "Fog")):
            macros[f"Eeightc{nm}Kitti"] = num(np.mean([e["kitti_by_condition"][cond] for e in ev]))
            macros[f"Eeightc{nm}Indomain"] = num(np.mean([e["in_domain_by_condition"][cond] for e in ev]))
        macros["EeightcEvalFrames"] = f"{sum(e['n_frames'] for e in ev):,}".replace(",", "{,}")
    r = load("ts_detector", "r4", "e8c_indomain_router.json")
    if not r:
        return
    m = r["oof_map"]
    for k, name in (("object", "Obj"), ("frame_cls", "Frame"), ("fixed_clear", "Fixed"), ("frame_pooled", "Pooled"),
                    ("geom_clocs", "Clocs")):
        macros[f"Eeightc{name}Map"] = num(m[k])
    c = r["primary_object_vs_frame_cls"]
    macros["EeightcDiff"] = f"{c['diff']:+.5f}"
    macros["EeightcLo"], macros["EeightcHi"] = f"{c['ci95'][0]:+.3f}", f"{c['ci95'][1]:+.3f}"
    macros["EeightcP"] = pc(c["p_one_sided"])
    macros["EeightcNScenes"] = str(c["n_scenes"])
    macros["EeightcAhead"] = str(c["scenes_ahead"])
    macros["EeightcBehind"] = str(c["scenes_behind"])
    macros["EeightcFolds"] = f"{r['folds_object_ahead_of_frame_cls']}/{len(r['per_fold'])}"
    macros["EeightcRefuted"] = "refuted" if r["primary_refuted"] else "not refuted"
    macros["EeightcNBoot"] = f"{r['n_boot_primary']:,}".replace(",", "{,}")
    rows_ = []
    for i, f in enumerate(r["per_fold"]):
        fm = f["map"]
        rows_.append([str(i + 1), str(f["n_test_scenes"]), num(fm["fixed_clear"]), num(fm["frame_cls"]), num(fm["object"]),
                      f"{fm['object'] - fm['frame_cls']:+.4f}"])
    macros["EeightcFoldRows"] = rows(rows_)
    macros["EeightcFixedDiff"] = f"{r['object_vs_fixed_clear']['diff']:+.3f}"
    macros["EeightcOrigDiff"] = macros.get("RtwoAOneDelta", "+0.064")
    q = r["object_vs_frame_pooled"]
    macros["EeightcPooledDiff"] = f"{q['diff']:+.3f}"
    macros["EeightcPooledP"] = pc(q["p_one_sided"])
    macros["EeightcPooledFolds"] = f"{r['folds_object_ahead_of_frame_pooled']}/{len(r['per_fold'])}"
    g = r["geom_clocs_vs_frame_cls"]
    macros["EeightcClocsDiff"] = f"{g['diff']:+.3f}"
    inc = r["object_vs_geom_clocs"]
    macros["EeightcIncDiff"] = f"{inc['diff']:+.3f}"
    macros["EeightcIncLo"], macros["EeightcIncHi"] = f"{inc['ci95'][0]:+.3f}", f"{inc['ci95'][1]:+.3f}"


def main():
    macros = {}

    e1 = load("exp1_harm", "exp1_harm.json")
    if e1:
        h1 = e1["H1"]
        macros["HarmRateAdverse"] = pct(h1["pooled_adverse_harm_rate"])
        macros["HarmRateAdverseCILo"] = pct(h1["ci95"][0])
        macros["HarmRateAdverseCIHi"] = pct(h1["ci95"][1])
        macros["HarmRateClear"] = pct(h1["clear_harm_rate"])
        macros["HarmExcess"] = f"{100 * h1['weather_excess_harm']:+.2f}"
        macros["HarmRefuted"] = "refuted" if h1["refuted"] else "not refuted"
        macros["HarmWeakFlag"] = "yes" if h1["weak_premise_flag"] else "no"
        macros["APHarmCells"] = str(h1["ap_level_harm_cells"])
        macros["Tau"] = num(e1["tau"])
        macros["ClearTrainFOne"] = num(e1["f1_clear_train"])
        macros["HarmMultimodal"] = pct(h1.get("harm_rate_multimodal_adverse"))

    e2 = load("exp2_routing", "exp2_routing.json")
    if e2:
        h2 = e2["H2"]["ADVERSE_POOLED"]
        macros["DeltaMapObjFrame"] = num(h2["map_object_minus_frame_cls"], 3)
        macros["PValObjFrame"] = num(h2["p_one_sided"], 3)
        macros["DeltaMapObjFeats"] = num(h2["map_object_minus_frame_feats"], 3)
        macros["DeltaMapObjConf"] = num(h2["map_object_minus_confidence"], 3)
        macros["AurocObj"] = num(h2["auroc_object"])
        macros["AurocFrameCls"] = num(h2["auroc_frame_cls"])
        macros["AurocConf"] = num(h2["auroc_confidence"])
        h2h = e2["H2"].get("HELDOUT", {})
        macros["DeltaMapObjFrameHeldout"] = num(h2h.get("map_object_minus_frame_cls"), 3)
        macros["HTwoRefuted"] = "refuted" if e2["H2"]["refuted"] else "not refuted"
        for arm, label in (("fixed_clear", "Fixed"), ("fixed", "FixedAll"), ("confidence", "Conf"),
                           ("frame_cls", "FrameCls"), ("frame_oracle", "FrameOracle"),
                           ("frame_feats", "FrameFeats"), ("object", "Object")):
            m = e2["map"].get(arm, {})
            macros[f"Map{label}Clear"] = num(m.get("clear0"))
            macros[f"Map{label}Adverse"] = num(m.get("ADVERSE_POOLED"))
            macros[f"Map{label}Heldout"] = num(m.get("HELDOUT"))
            macros[f"NParams{label}"] = thou(e2["n_params"].get(arm, "?"))
        if e2.get("weather_classifier"):
            macros["WeatherClfAcc"] = num(e2["weather_classifier"]["train_conditions_test_acc"])

    e3 = load("exp3_ablation", "exp3_ablation.json")
    if e3:
        full = e3.get("full", {})
        macros["AblFullAdverse"] = num(full.get("ADVERSE_POOLED"))
        for k in ("no_context", "-geometry", "-lidar_probes", "-camera_probes", "-radar_probes",
                  "-det_feats", "-conflict", "-confidence", "-frame_feats"):
            if k in e3:
                key = "Abl" + k.replace("-", "No").replace("_", "").title().replace(" ", "")
                macros[key.replace("-", "")] = num(full.get("ADVERSE_POOLED", 0) - e3[k].get("ADVERSE_POOLED", 0), 3)

    e4 = load("exp4_conformal", "exp4_conformal.json")
    if e4:
        h3 = e4["H3"]
        macros["SnowCovVanilla"] = num(h3["snow_vanilla_coverage"])
        macros["SnowCovConflict"] = num(h3["snow_conflict_coverage"])
        macros["SnowFNR"] = num(h3["snow_fnr_at_alpha_0.1"])
        macros["HThreeRefuted"] = "refuted" if h3["refuted"] else "not refuted"
        macros["ClearRadiusRatio"] = num(h3.get("clear_radius_ratio_conflict_over_vanilla"))
        obj_loc = e4.get("object", {}).get("loc", {}).get("0.1", {})
        fog3 = obj_loc.get("fog3", {})
        if fog3:
            macros["FogSevereCovVanilla"] = num(fog3.get("vanilla", {}).get("coverage"))
        obj_crc = e4.get("object", {}).get("crc", {}).get("0.1", {})
        if "fog3" in obj_crc:
            macros["FogSevereFNR"] = num(obj_crc["fog3"].get("fnr"))

    e5 = load("exp5_cadc", "exp5_cadc.json")
    if e5 and "map" in e5:
        macros["CadcMapObj"] = num(e5["map"]["object"]["ALL"])
        macros["CadcMapObjCILo"] = num(e5["map"]["object"]["ALL_ci"][0])
        macros["CadcMapObjCIHi"] = num(e5["map"]["object"]["ALL_ci"][1])
        macros["CadcMapFrameCls"] = num(e5["map"]["frame_cls"]["ALL"])
        macros["CadcMapFixed"] = num(e5["map"]["fixed_clear"]["ALL"])
        macros["CadcDeltaObjFrame"] = num(e5["object_minus_frame_cls"]["diff"], 3)
        macros["CadcPValObjFrame"] = num(e5["object_minus_frame_cls"]["p_one_sided"], 3)
        macros["CadcAll"] = num(e5["ap_level"]["all"])
        macros["CadcMinusCamera"] = num(e5["ap_level"]["minus_camera"])
        macros["CadcMinusLidar"] = num(e5["ap_level"]["minus_lidar"])
        hs = e5["harm"]
        macros["CadcHarmPooled"] = pct(hs["pooled"]["harm_rate"])
        macros["CadcHarmPooledCILo"] = pct(hs["pooled"]["harm_ci"][0])
        macros["CadcHarmPooledCIHi"] = pct(hs["pooled"]["harm_ci"][1])
        macros["CadcHarmCamera"] = pct(hs["camera"]["harm_rate"])
        macros["CadcHelpCamera"] = pct(hs["camera"]["help_rate"])
        macros["CadcNFrames"] = str(e5["n_frames"])
        macros["CadcNCars"] = str(e5["n_gt_cars"])

    e7 = load("exp7_dropout", "exp7_dropout.json")
    if e7:
        for tag in ("clear0", "fog3", "snow3", "night2"):
            if tag in e7.get("object", {}):
                r = e7["object"][tag]
                t = tag.title().replace("0", "").replace("2", "").replace("3", "")
                macros[f"Drop{t}All"] = num(r["none"])
                macros[f"Drop{t}NoCam"] = num(r["no_camera"])
                macros[f"Drop{t}NoLid"] = num(r["no_lidar"])
                macros[f"Drop{t}NoRad"] = num(r["no_radar"])

    # Full-data 5-seed multiseed is a long-running check (~4h/seed); if it has finished, prefer it.
    # Otherwise fall back to the dev-subset (1,654-frame) 5-seed run, explicitly labelled as such in the
    # manuscript (macro name carries "Dev") so the two are never conflated.
    ms_full = load("multiseed_exp2.json")
    prefix, ms = ("", ms_full) if ms_full else ("Dev", load("multiseed_exp2_devrun.json"))
    if ms:
        for arm, lab in (("object", ""), ("frame_cls", "FrameCls"), ("fixed_clear", "Fixed")):
            d = ms["map_mean_sd"][arm]
            macros[f"Seed{prefix}{lab}AdverseMean"] = num(d["ADVERSE_POOLED"][0])
            macros[f"Seed{prefix}{lab}AdverseSd"] = num(d["ADVERSE_POOLED"][1], 3)
            macros[f"Seed{prefix}{lab}HeldoutMean"] = num(d["HELDOUT"][0])
            macros[f"Seed{prefix}{lab}HeldoutSd"] = num(d["HELDOUT"][1], 3)
        dl = ms.get("delta_object_minus_frame_cls_adverse")
        if dl:
            macros[f"Seed{prefix}DeltaMin"] = num(min(dl["per_seed"]), 3)
            macros[f"Seed{prefix}DeltaMax"] = num(max(dl["per_seed"]), 3)
        macros["SeedIsFullData"] = "no" if prefix else "yes"

    s1 = load("supplementary", "s1_scene_bootstrap.json")
    if s1:
        adv = s1.get("ADVERSE_POOLED", {})
        ovf = adv.get("object_vs_frame_cls", {})
        macros["SceneDeltaObjFrame"] = num(ovf.get("diff"), 3)
        macros["SceneCILo"] = num(ovf.get("ci95", [None, None])[0], 3)
        macros["SceneCIHi"] = num(ovf.get("ci95", [None, None])[1], 3)
        macros["ScenePVal"] = num(ovf.get("p_one_sided"), 3)
        macros["SceneWins"] = esc(ovf.get("scenes_won"))
        fvf = adv.get("frame_cls_vs_fixed_clear", {})
        macros["SceneDeltaFrameFixed"] = num(fvf.get("diff"), 3)
        macros["SceneFrameFixedP"] = num(fvf.get("p_one_sided"), 3)

    s2 = load("supplementary", "s2_decomposition.json")
    if s2:
        adv = s2.get("ADVERSE_POOLED", {})
        macros["ShareFromScores"] = f"{100 * adv.get('share_from_scores', 0):.0f}"
        macros["ShareFromPositions"] = f"{100 * adv.get('share_from_positions', 0):.0f}"

    s3 = load("supplementary", "s3_controls.json")
    if s3:
        clocs = s3.get("CLOCs-style (conf + conflict + det feats, no probes)", {})
        macros["ClocsAdverse"] = num(clocs.get("ADVERSE_POOLED"))
        macros["ClocsParams"] = thou(clocs.get("params", "?"))
        obj = s3.get("per-object (ours)", {})
        macros["SThreeObjAdverse"] = num(obj.get("ADVERSE_POOLED"))
        noabs = s3.get("per-object, negative-evidence term removed (test time)", {})
        macros["NoAbsAdverse"] = num(noabs.get("ADVERSE_POOLED"))
        probes_only = s3.get("probes only (no conf / conflict / det feats)", {})
        macros["ProbesOnlyAdverse"] = num(probes_only.get("ADVERSE_POOLED"))
        h88 = s3.get("per-frame features, hidden 88 (≈ ours' params)", {})
        macros["FrameFeatsWideAdverse"] = num(h88.get("ADVERSE_POOLED"))
        h88c = s3.get("per-frame classifier, hidden 88", {})
        macros["FrameClsWideAdverse"] = num(h88c.get("ADVERSE_POOLED"))
        h40 = s3.get("per-object, hidden 40 (≈ per-frame params)", {})
        macros["ObjNarrowAdverse"] = num(h40.get("ADVERSE_POOLED"))

    s4 = load("supplementary", "s4_noradar.json")
    if s4:
        b = s4.get("boreas", {})
        macros["NoRadarObjAdverse"] = num(b.get("object", {}).get("ADVERSE_POOLED"))
        macros["NoRadarFrameClsAdverse"] = num(b.get("frame_cls", {}).get("ADVERSE_POOLED"))
        macros["NoRadarFixedAdverse"] = num(b.get("fixed_clear", {}).get("ADVERSE_POOLED"))
        h2 = s4.get("boreas_H2", {}).get("ADVERSE_POOLED", {})
        macros["NoRadarDelta"] = num(h2.get("diff"), 3)
        macros["NoRadarPFrame"] = num(h2.get("p_frame"), 3)
        macros["NoRadarPScene"] = num(h2.get("p_scene"), 3)
        c = s4.get("cadc", {})
        if "object" in c:
            macros["NoRadarCadcObj"] = num(c["object"]["mAP"])
            macros["NoRadarCadcFrameCls"] = num(c.get("frame_cls", {}).get("mAP"))
            ovf = c.get("object_vs_frame_cls", {})
            macros["NoRadarCadcDelta"] = num(ovf.get("diff"), 3)
            macros["NoRadarCadcPDrive"] = num(ovf.get("p_drive_cluster"), 3)

    s5 = load("supplementary", "s5_baselines.json")
    if s5:
        adv = s5.get("ADVERSE_POOLED", {})
        macros["NaiveMaxAdverse"] = num(adv.get("untrained max"))
        macros["NaiveNoisyOrAdverse"] = num(adv.get("untrained noisy_or"))
        macros["LidarOnlyAdverse"] = num(adv.get("lidar only (raw detector)"))
        macros["CameraOnlyAdverse"] = num(adv.get("camera only (raw detector)"))
        macros["RadarOnlyAdverse"] = num(adv.get("radar only (raw detector)"))
        macros["FittedFixedAdverse"] = num(adv.get("fitted fixed (status quo)"))

    sim2real = R / "sim2real" / "sim2real.json"
    if sim2real.exists():
        macros["SimToRealDone"] = "yes"

    manifest = load("cache", "boreas", "manifest.json")
    if manifest:
        macros["NFramesFull"] = thou(manifest.get("n_frames"))
        sp = manifest.get("splits", {})
        macros["NTrainFrames"] = str(sp.get("train", "?"))
        macros["NCalFrames"] = str(sp.get("cal", "?"))
        macros["NTestFrames"] = str(sp.get("test", "?"))
        sc = manifest.get("scenes", {})
        macros["NTrainScenes"] = str(len(sc.get("train", [])))
        macros["NCalScenes"] = str(len(sc.get("cal", [])))
        macros["NTestScenes"] = str(len(sc.get("test", [])))

    r2_macros(macros)
    r3_macros(macros)
    r4_macros(macros)
    e10_macros(macros)
    e11_macros(macros)
    e12_macros(macros)
    e13_macros(macros)
    e15_macros(macros)
    e16_macros(macros)
    e17_macros(macros)
    e8c_macros(macros)

    lines = ["% AUTO-GENERATED by paper/latex/make_paper_data.py — do not edit by hand.",
             "% Regenerate after any change to results/*.json. \\DefNum (main.tex) unconditionally overwrites",
             "% whatever autogen_stubs.tex defined.", ""]
    import re
    for k, v in sorted(macros.items()):
        v = re.sub(r"(?<![\w\-])-(?=\d)", "$-$", str(v))  # true minus sign for every negative number
        lines.append(f"\\DefNum{{{k}}}{{{v}}}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    missing = [k for k, v in macros.items() if v in ("?", None)]
    print(f"wrote {len(macros)} macros to {OUT}" + (f"; {len(missing)} UNRESOLVED: {missing}" if missing else ""))
    stage_figures()


if __name__ == "__main__":
    main()
