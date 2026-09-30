# Paper 02 — Conflict-Aware Late Fusion (implementation)

Which sensor to believe when weather degrades them differently: a per-object, degradation-conditioned
arbitration head over **frozen** camera / LiDAR / radar detectors, with conformal guarantees.
Plan: [PLAN.md](PLAN.md) · pre-registration: [notes/falsification.md](notes/falsification.md) ·
data caveats: [notes/data_reality.md](notes/data_reality.md) · detectors: [notes/detectors.md](notes/detectors.md) ·
RAF positioning: [notes/related_work_positioning.md](notes/related_work_positioning.md)

## Environment

```powershell
$py = "F:\CLAUDE\AI2\shared\envs\unitraj\Scripts\python.exe"    # Python 3.10, torch 2.0.1+cu118, numpy 1.24
& $py -m pip install --no-deps asrl-pylgmath asrl-pysteam       # pyboreas deps, no numpy/torch churn
& $py -m pip install --no-deps -e code\pyboreas
$env:TORCH_HOME = "F:\CLAUDE\AI2\paper02-conflict-fusion\code\checkpoints\torch"
```
Run every command from `paper02-conflict-fusion/`.

## Pipeline

| step | command | time |
|---|---|---|
| 1. data (labelled frames only) | `& $py src\fetch_boreas.py --phase A` then `--phase B` | ~1 h, 41 GB |
| 2. validate LiDAR port | `& $py -m src.detectors.pointpillars --selfcheck` | 5 min |
| 3. look at the simulator | `& $py -m src.viz_weather --frame-index 300` → `results/figures/weather_panel.png` | 1 min |
| 4. **frozen-detector cache (once)** | `& $py -m src.cache_detections --workers 3 --threads 5 [--stride 3]` | dev (stride 3): ~6 h CPU; full: ~3.5 days CPU / ~1 day GPU |
| 5. candidate tables | `& $py -m src.candidates` | minutes |
| 6. experiments 1–8 + report | `& $py -m src.run_experiments --exp all` → `results/REPORT.md` | ~30 min CPU |
| 7. sim-to-real check | `& $py -m src.sim2real --sequence boreas-2021-01-26-11-22 --n 60` | ~15 min |
| 8. CADC (public, no registration) | `& $py srcetch_cadc.py` -> `& $py -m src.cadc_cache` -> `& $py -m src.candidates --dataset cadc` -> `--exp 5` | hours |
| 9. MAN TruckScenes v1.2 (public S3) | `& $py srcetch_truckscenes.py` -> `& $py -m src.ts_cache --workers 4 --threads 4 --device cpu` -> `& $py -m src.candidates --dataset truckscenes` | ~1 day CPU |
| 10. Registration-2 experiments | `& $py -m src.r2_experiments` (baselines, ablations, conformal repair, CV, range/latency); `& $py -m src.r2d_cadc --build` then `& $py -m src.r2d_cadc`; `& $py -m src.r2a_truckscenes` | hours |
| 11. paper numbers + figures | `& $py -m src.paper_figures --all`; `& $py paper\latex\make_paper_data.py`; then `pdflatex`/`bibtex` in `paper/latex` | minutes |
| 10b. Registration 3 (closing the gap to the set-transformer) | `& $py -m src.r3_experiments dev` (development split only), then `& $py -m src.r3_experiments final --variant V2 --part truckscenes` / `--part boreas` / `--part cv` | ~1.5 h |
| tests | `& $py -m pytest tests -q` | ~6 min |

`cache_detections` is resumable and ordered (clear0, snow2, fog2, rain2, night2, snow3, …) so a partial cache is
already usable: `src.candidates` builds tables for whatever conditions exist. Delete `results/models/` (or pass
`--retrain`) after the cache grows, because fitted heads are cached there.

## Layout

```
src/
  fetch_boreas.py        selective S3 download (s5cmd)
  boreas_data.py         frame index (LiDAR-anchored, pose-compensated), loaders, GT + don't-care
  weather.py             graded fog / rain / snow / night on raw LiDAR, camera, radar
  detectors/             pointpillars.py (pure-torch mmdet3d port), camera_frcnn.py (+ mono lift), radar_cfar.py
  degradation.py         per-object probes + frame features
  cache_detections.py    THE offline pass (run once)
  candidates.py          cross-modal association, conflict features, labels
  arbitrate.py           fusion rule + routers: fixed / confidence / per-frame (classifier, oracle, features) / per-object
  harm.py                counterfactual harm labels and rates (H1)
  conformal.py           split conformal, conflict-inflated scores, Mondrian, weighted (shift), CRC (H3)
  evaluate.py, metrics.py  centre-distance AP, decisions, bootstrap
  run_experiments.py     Exp 1–8, tables, figures, REPORT.md
  sim2real.py            unlabelled real-snow Boreas vs simulated severities
  cadc_cache.py          CADC external validation cache (camera + LiDAR)
  synthetic.py           synthetic cache for tests only — never a result
tests/                   unit + end-to-end synthetic pipeline
configs/default.yaml     every parameter
```

## Citation and licence

See `CITATION.cff`. Code: MIT (`LICENSE`). Datasets are **not** redistributed: Boreas, CADC and MAN TruckScenes
(CC BY-NC-SA 4.0) must be obtained from their owners; derived aggregate results in `results/` inherit that non-commercial term.
Pre-registrations: `notes/falsification.md` (R1), `notes/registration2.md` (R2); outcomes: `notes/r2_outcomes.md`.
