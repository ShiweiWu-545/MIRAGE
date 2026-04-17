#!/usr/bin/env bash
set -euo pipefail

LABEL="${1:-test1_DG_blind}"
python run_train/fold0_model.py "$LABEL"
