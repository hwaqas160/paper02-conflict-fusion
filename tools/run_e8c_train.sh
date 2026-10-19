#!/bin/bash
# E8-C: train both cross-fitted detectors, sequentially. Resumable: re-run this script after any interruption.
cd "$(dirname "$0")/.."
PY="F:/CLAUDE/AI2/shared/envs/unitraj/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
for FOLD in 0 1; do
  if [ ! -f results/ts_detector/fold${FOLD}_final.pt ]; then
    "$PY" -m src.ts_detector train --fold $FOLD --epochs 4 --per-epoch 3000 --batch 2 --workers 6 --lr 2e-4 >> logs/tsdet_fold${FOLD}.log 2>&1
  fi
done
echo ALL_DONE >> logs/tsdet_fold1.log
