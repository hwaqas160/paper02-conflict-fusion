"""Extract every number the paper cites into paper/latex/autogen_numbers.tex as \\newcommand macros.

Run AFTER the full-resolution run: src.candidates -> src.run_experiments --exp all --retrain -> multiseed
(seeds 0-4) -> src.supplementary. No number in sections/*.tex should be typed by hand; it should be a
\\NumXxx{} macro defined here, generated straight from the JSON the experiments wrote. This is the
mechanical safeguard against a transcription mismatch between the paper and results/*.json.

    python paper/latex/make_paper_data.py
"""
import glob
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = Path(__file__).resolve().parent / "autogen_numbers.tex"
FIGDIR = Path(__file__).resolve().parent / "figures"

FIGURE_MAP = {
    "figures/weather_panel.png": "weather_panel.png",
    "exp1_harm/harm_by_condition.png": "harm_by_condition.png",
    "exp2_routing/map_by_condition.png": "map_by_condition.png",
    "exp4_conformal/coverage_transfer.png": "coverage_transfer.png",
    "exp7_dropout/dropout.png": "dropout.png",
    "supplementary/forest_headline.png": "forest_headline.png",
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
    qp = R / "exp8_qualitative" / "qualitative_cases.png"
    if qp.exists():
        shutil.copyfile(qp, FIGDIR / "qual_pair.png")
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


def pct(x, d=2):
    return "?" if x is None else f"{100 * x:.{d}f}"


def num(x, d=3):
    return "?" if x is None else f"{x:.{d}f}"


def esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%")


TEX_LABEL = {"object": "Per-object router (ours)", "fixed_clear": "Fixed weights (status quo)",
             "frame_cls": "Per-frame weather classifier", "proben": "ProbEn \\citep{chen2022proben}",
             "dempster_shafer": "Dempster--Shafer evidential \\citep{shafer1976}",
             "clocs": "CLOCs \\citep{pang2020clocs}", "set_transformer": "Set-transformer fusion \\citep{lee2019settransformer}",
             "moe_frame": "Per-frame mixture of experts \\citep{mees2016moe}", "oracle": "Oracle selector (upper bound)",
             "object_noradar": "Per-object router, radar removed"}
ABL_LABEL = {"no_absence_term": "negative-evidence (absence) term", "no_learned_precision": "learned position precisions",
             "no_harm_aux_loss": "harm auxiliary loss", "no_localisation_loss": "localisation loss",
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
            out.append([TEX_LABEL[a], str(b["params"][a]), num(m.get("clear0")), num(m.get("ADVERSE_POOLED")),
                        num(m.get("HELDOUT")), num(m.get("fog3")), num(b.get("cadc", {}).get(a)),
                        cmp_cell(v) if v else "--"])
        macros["RtwoBRows"] = rows(out)
        macros["RtwoBHeadroom"] = f"{100 * b['headroom_captured']:.0f}"
        camel = {"set_transformer": "SetTransformer", "clocs": "Clocs", "proben": "ProbEn",
                 "dempster_shafer": "DempsterShafer", "moe_frame": "MoeFrame", "frame_cls": "FrameCls",
                 "fixed_clear": "FixedClear"}
        for a, v in b["vs_ours"].items():
            suf = camel.get(a, a.title().replace("_", ""))
            macros[f"RtwoBDiff{suf}"] = num(v["diff"], 4)
            macros[f"RtwoBLo{suf}"] = num(v["ci95"][0], 4)
            macros[f"RtwoBHi{suf}"] = num(v["ci95"][1], 4)
        best = max((a for a in b["vs_ours"]), key=lambda a: b["map"][a]["ADVERSE_POOLED"])
        macros["RtwoBBestName"] = TEX_LABEL[best].split(" \\cite")[0]
        macros["RtwoBBestGap"] = num(b["vs_ours"][best]["diff"], 3)
        macros["RtwoBBestP"] = pc(b["vs_ours"][best]["p_holm"])
        macros["RtwoBNBeaten"] = str(sum(v["diff"] > 0 and v["p_holm"] < 0.05 for v in b["vs_ours"].values()))
        macros["RtwoBNCompared"] = str(len(b["vs_ours"]))
        macros["RtwoBOracleAdverse"] = num(b["map"]["oracle"]["ADVERSE_POOLED"])
    c = load(*r2, "r2c_ablation.json")
    if c:
        out = [["full router (ours)", num(c["map"]["full"]["clear0"]), num(c["map"]["full"]["ADVERSE_POOLED"]), "--", "--"]]
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
        macros["RtwoFDelta"], macros["RtwoFP"] = num(o["diff"], 4), pc(o["p_one_sided"])
        macros["RtwoFCILo"], macros["RtwoFCIHi"] = num(o["ci95"][0], 4), num(o["ci95"][1], 4)
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
            macros[f"RtwoA{lab}Delta"] = num(a[key]["diff"], 4)
            macros[f"RtwoA{lab}P"] = pc(a[key]["p_one_sided"])
            macros[f"RtwoA{lab}CILo"], macros[f"RtwoA{lab}CIHi"] = num(a[key]["ci95"][0], 4), num(a[key]["ci95"][1], 4)
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
                both = [["full router (ours)", num(c["map"]["full"]["clear0"]), num(c["map"]["full"]["ADVERSE_POOLED"]), "--", "--",
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
        macros["RtwoAOffDelta"], macros["RtwoAOffP"] = num(o["A1"]["diff"], 4), pc(o["A1"]["p_one_sided"])
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
               ["zero-shot, resolution-normalised"] + [num(d["D1_normalised"]["map"][x]) for x in arms] +
               [f"{d['D1_normalised']['object_vs_frame_cls']['diff']:+.4f} ({pv(d['D1_normalised']['object_vs_frame_cls']['p_one_sided'])})"]]
        for k, v in d["D2"].items():
            o = v["object_vs_frame_cls"]
            out.append([f"few-shot, $k={k}$ drives"] + [num(v["pooled_map"][x]) for x in arms] +
                       [f"{o['diff']:+.4f} ({pv(o['p_one_sided'])})"])
        macros["RtwoDRows"] = rows(out)
        o = d["D1_normalised"]["object_vs_frame_cls"]
        macros["RtwoDOneDelta"], macros["RtwoDOneP"] = num(o["diff"], 4), pc(o["p_one_sided"])
        macros["RtwoDOneRefuted"] = "refuted" if d["D1_refuted"] else "not refuted"
        kmax = str(max(int(k) for k in d["D2"]))
        o = d["D2"][kmax]["object_vs_frame_cls"]
        macros["RtwoDTwoDelta"], macros["RtwoDTwoP"] = num(o["diff"], 4), pc(o["p_one_sided"])
        macros["RtwoDTwoRefuted"] = "refuted" if d["D2_refuted"] else "not refuted"
        macros["RtwoDTwoMonotone"] = "yes" if d["D2_monotone"] else "no"
        macros["RtwoDNDrives"] = str(d["n_drives"])


R3_LABEL = {"original": "Original router", "V0": "Original router, re-fit", "V1": "V1: attention context",
            "V2": "V2: bounded correction (final)", "V3": "V3: V1 + V2", "set_transformer": "Set-transformer (black box)",
            "set_noattn": "Set-transformer, attention removed", "variant_rule_only": "V2, correction switched off"}


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
        macros["RthreeBoVsOrigLo"] = num(b["vs"]["original"]["ci95"][0], 4)
        macros["RthreeBoVsOrigHi"] = num(b["vs"]["original"]["ci95"][1], 4)
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
        macros["DeltaMapObjFrame"] = num(h2["map_object_minus_frame_cls"], 4)
        macros["PValObjFrame"] = num(h2["p_one_sided"], 3)
        macros["DeltaMapObjFeats"] = num(h2["map_object_minus_frame_feats"], 4)
        macros["DeltaMapObjConf"] = num(h2["map_object_minus_confidence"], 4)
        macros["AurocObj"] = num(h2["auroc_object"])
        macros["AurocFrameCls"] = num(h2["auroc_frame_cls"])
        macros["AurocConf"] = num(h2["auroc_confidence"])
        h2h = e2["H2"].get("HELDOUT", {})
        macros["DeltaMapObjFrameHeldout"] = num(h2h.get("map_object_minus_frame_cls"), 4)
        macros["HTwoRefuted"] = "refuted" if e2["H2"]["refuted"] else "not refuted"
        for arm, label in (("fixed_clear", "Fixed"), ("fixed", "FixedAll"), ("confidence", "Conf"),
                           ("frame_cls", "FrameCls"), ("frame_oracle", "FrameOracle"),
                           ("frame_feats", "FrameFeats"), ("object", "Object")):
            m = e2["map"].get(arm, {})
            macros[f"Map{label}Clear"] = num(m.get("clear0"))
            macros[f"Map{label}Adverse"] = num(m.get("ADVERSE_POOLED"))
            macros[f"Map{label}Heldout"] = num(m.get("HELDOUT"))
            macros[f"NParams{label}"] = str(e2["n_params"].get(arm, "?"))
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
                macros[key.replace("-", "")] = num(full.get("ADVERSE_POOLED", 0) - e3[k].get("ADVERSE_POOLED", 0), 4)

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
        macros["CadcDeltaObjFrame"] = num(e5["object_minus_frame_cls"]["diff"], 4)
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
        macros["SceneDeltaObjFrame"] = num(ovf.get("diff"), 4)
        macros["SceneCILo"] = num(ovf.get("ci95", [None, None])[0], 4)
        macros["SceneCIHi"] = num(ovf.get("ci95", [None, None])[1], 4)
        macros["ScenePVal"] = num(ovf.get("p_one_sided"), 3)
        macros["SceneWins"] = esc(ovf.get("scenes_won"))
        fvf = adv.get("frame_cls_vs_fixed_clear", {})
        macros["SceneDeltaFrameFixed"] = num(fvf.get("diff"), 4)
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
        macros["ClocsParams"] = str(clocs.get("params", "?"))
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
        macros["NoRadarDelta"] = num(h2.get("diff"), 4)
        macros["NoRadarPFrame"] = num(h2.get("p_frame"), 3)
        macros["NoRadarPScene"] = num(h2.get("p_scene"), 3)
        c = s4.get("cadc", {})
        if "object" in c:
            macros["NoRadarCadcObj"] = num(c["object"]["mAP"])
            macros["NoRadarCadcFrameCls"] = num(c.get("frame_cls", {}).get("mAP"))
            ovf = c.get("object_vs_frame_cls", {})
            macros["NoRadarCadcDelta"] = num(ovf.get("diff"), 4)
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
        macros["NFramesFull"] = str(manifest.get("n_frames"))
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
