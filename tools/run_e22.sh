#!/bin/bash
cd "$(dirname "$0")/.."
PY="F:/CLAUDE/AI2/shared/envs/unitraj/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
until grep -q -E "E21_DONE" logs/e21_after.log; do sleep 120; done
for s in fit probes candidates router; do "$PY" -m src.e22_combined $s || exit 1; done
echo E22_DONE
