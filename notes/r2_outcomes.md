# Registration 2 -- outcomes so far (2026-09-24). Reported as found, refutations included.

| id | verdict | evidence |
|---|---|---|
| R2-B1 (Boreas) | **partly refuted** | ours 0.777 adverse; beats CLOCs, ProbEn, D--S, MoE, per-frame, fixed (all p_Holm 0.007); **ties** the 72k-param set-transformer (Δ 0.0005, p_Holm 0.39). On CADC zero-shot (no radar) CLOCs 0.763 > ours 0.734. Oracle 0.847; ours captures 56 % of the headroom. |
| R2-C1 | **only 1 of 6 mechanisms contributes** | harm auxiliary loss (+0.0066, p_Holm 0.03). Absence term, sensor dropout, frame context: no measurable adverse-mAP gain. Learned precisions and localisation loss give *identical* numbers (precisions are trained only through the localisation loss, so the two ablations are the same intervention) and removal slightly *helps* (+0.005). |
| R2-D1 | **refuted** | normalised probes change (ring ratio 0.03 -> 0.13, log-counts +1.2) but router scores move by 1e-4: the router is nearly invariant to those LiDAR probes, so resolution does not explain the CADC gap. Δ -0.0047, p 0.97. |
| R2-D2 | **refuted** | Δ(k=1,5,10) = +0.001, +0.018 (p 0.001), -0.002; not monotone, k=10 not > 0. |
| R2-E1 | **refuted** as registered | fog3 localisation coverage 0.915 (repaired) but frame FNR 0.379 > 0.15; FNR floor with every detection accepted is 0.30 (detector-limited). Exploratory: label-free support check flags 100 % of fog3 / 0 % of clear + snow; unflagged severe frames FNR 0.088. |
| R2-F1 | **supported** | 5/5 folds, out-of-fold Δ 0.086 [0.068, 0.108], p 0.001. |
| R2-G | descriptive | gain grows with range: 40-70 m AP 0.671 vs 0.556 (status quo); <30 m miss 12.5 % vs 19.9 %; 2.9 ms/frame CPU. |

## Consequences for the claims (do not overstate)
* The defensible contribution is the **per-object vs per-frame** result and the harm-aware, range-dependent gain, replicated by CV.
* "Every component matters" is **false** and must not be claimed; report the ablation as it is and shrink the method description to what earns its place.
* Against a black-box set-transformer the router is *equal*, at 1/4 the parameters and with an interpretable fusion rule -- claim parity + interpretability, not superiority.
* Pending: TruckScenes (real weather + 4D radar) decides whether the story generalises (R2-A).
