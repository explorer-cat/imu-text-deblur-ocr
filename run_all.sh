#!/bin/bash
# Full pipeline: train both models, restore test set, build the final table.
#   bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=./.venv/bin/python
CFG=configs/default.yaml

echo "== [3] train baseline =="
$PY train.py --config $CFG --model baseline
echo "== [4] train imu =="
$PY train.py --config $CFG --model imu
echo "== [5a] restore =="
$PY infer.py --config $CFG --model baseline
$PY infer.py --config $CFG --model imu
echo "== [5b] final table =="
$PY make_table.py --config $CFG
echo "== DONE =="
