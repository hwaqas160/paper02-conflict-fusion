#!/bin/bash
# waits for camera training, then detect -> build -> candidates -> router (resumable; re-run after interruption)
cd "$(dirname "$0")/.."
PY="F:/CLAUDE/AI2/shared/envs/unitraj/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
until [ -f results/ts_detector/camera/cam_fold1_final.pt ]; do sleep 120; done
"$PY" -m src.e21_camera_train detect --workers 6 --threads 4
"$PY" -m src.e21_camera_train build --workers 8
"$PY" -m src.e21_finish candidates
"$PY" -m src.e21_finish router
echo E21_DONE
