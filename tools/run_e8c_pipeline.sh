#!/bin/bash
# E8-C after training: detect (CPU) -> build cache -> candidates -> router CV. Resumable: re-run after any interruption.
cd "$(dirname "$0")/.."
PY="F:/CLAUDE/AI2/shared/envs/unitraj/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
TOTAL=23902
n() { ls results/ts_detector/dets 2>/dev/null | grep -c '^[^.]*\.npz$'; }
if [ "$(n)" -lt "$TOTAL" ]; then
  "$PY" -m src.e8c_rebuild detect --device cpu --workers 6 --threads 4 >> logs/e8c_detect.log 2>&1
fi
"$PY" -m src.e8c_rebuild build --workers 10 --threads 2 >> logs/e8c_build.log 2>&1
"$PY" -m src.e8c_rebuild candidates >> logs/e8c_candidates.log 2>&1
"$PY" -m src.e8c_router 10000 1000 >> logs/e8c_router.log 2>&1
echo PIPELINE_DONE >> logs/e8c_router.log
