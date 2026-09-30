# Data reality check — what the datasets can and cannot support (2026-09-15)

## Finding: Boreas has no labelled adverse weather

PLAN.md §9 assumed "calibrate on clear-weather Boreas sequences, deploy on snow sequences" with ground truth.
That is not possible:

* All public 3D boxes are in **one** sequence, `boreas-objects-v1`, recorded **2020-09-01** — "obtained in sunny
  weather" (Burnett et al., IJRR 2023, §8). 4,951 labelled LiDAR frames are public (37 scenes); the 16 test
  scenes' labels are withheld by the benchmark.
* Snow sequences (e.g. `boreas-2021-01-26-10-59`, `-11-22`) have sensor data and poses but no boxes.

## Consequence for the design (decided, not asked — revisit if you disagree)

| need | source |
|---|---|
| exact GT under *graded, modality-specific* degradation | Boreas-objects-v1 + physics-based simulation (`src/weather.py`): fog, rain, snow × 3 severities, night |
| labelled **real** snow | CADC (camera + LiDAR, no radar) — Exp 5, registration required |
| evidence the simulator is not fantasy | `src/sim2real.py`: frame-level degradation statistics of real unlabelled Boreas snow vs simulated severities |
| per-frame weather classifier baseline | trained on simulated conditions (frame-level features only) |

This changes the wording of H3 in the paper from "clear → real snow" to "clear → graded simulated snow, with
real-snow external validation on CADC". The pre-registration (`notes/falsification.md`) was written with this
design, before any result.

**Why this still serves the Canada application (PLAN §6):** the work extends Boreas' *own* labelled benchmark to
adverse weather and uses Boreas' real snow sequences for validation; CADC remains the Waterloo real-snow cohort.

## Selective download (actual)

| item | size |
|---|---|
| calib, applanix, labels_detection, labels_tracking | ~93 MB |
| phase A: every 5th labelled frame (991 frames): LiDAR + nearest camera + nearest & previous radar | ~8.6 GB |
| phase B: remaining 3,960 labelled frames | ~33 GB (running) |
| full `boreas-objects-v1` if pulled naively | 142 GB |

Tool: `shared/tools/s5cmd.exe` (no AWS CLI needed). Script: `src/fetch_boreas.py`.
