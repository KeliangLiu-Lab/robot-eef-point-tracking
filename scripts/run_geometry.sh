#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_ROOT"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${EEF_DATA_ROOT:-$PACKAGE_ROOT/data/lerobot}"
CALIBRATION_ROOT="${EEF_CALIBRATION_ROOT:-$PACKAGE_ROOT/configs/calibration_template}"
GEOMETRY_ROOT="${EEF_GEOMETRY_ROOT:-$PACKAGE_ROOT/outputs/eef_tracks_calibrated_v5_geometry}"
V5_GEOMETRY_ROOT="${EEF_V5_GEOMETRY_ROOT:-$PACKAGE_ROOT/outputs/eef_tracks_calibrated_v5_geometry_final}"
FINAL_ROOT="${EEF_FINAL_ROOT:-$PACKAGE_ROOT/outputs/eef_tracks_calibrated_v5}"
AGILEX_DATASET="agilex7000_manip_short30_rot6d_rowmajor_h32_v4_prompt_reviewed"
WORKERS="${WORKERS:-16}"

mkdir -p "$GEOMETRY_ROOT"

"$PYTHON" src/batch_project_agx_geometry.py \
  --data-root "$DATA_ROOT" \
  --calibration-root "$CALIBRATION_ROOT" \
  --output-root "$GEOMETRY_ROOT" \
  --final-output-root "$FINAL_ROOT" \
  --datasets cup white color ordered \
  --workers "$WORKERS"

EEF_AGILEX7000_DATASET="$DATA_ROOT/$AGILEX_DATASET" \
EEF_GEOMETRY_OUTPUT_ROOT="$GEOMETRY_ROOT" \
  "$PYTHON" src/batch_project_agilex7000_geometry.py \
    --dataset-root "$DATA_ROOT/$AGILEX_DATASET" \
    --workers "$WORKERS"

"$PYTHON" src/backfill_agilex7000_shared_geometry.py \
  --dataset-root "$DATA_ROOT/$AGILEX_DATASET" \
  --output-root "$GEOMETRY_ROOT/$AGILEX_DATASET" \
  --override configs/overrides/agilex7000_hanging_shared_right_base_invariant.json \
  --workers "${BACKFILL_WORKERS:-8}" \
  --publish-manifest

"$PYTHON" src/upgrade_geometry_v5.py \
  --input-root "$GEOMETRY_ROOT" \
  --output-root "$V5_GEOMETRY_ROOT" \
  --final-root "$FINAL_ROOT" \
  --override configs/overrides/agilex7000_ep1952_e217.json \
  --override configs/overrides/agilex7000_ep2829_source44.json \
  --workers "$WORKERS"

echo "Geometry stage complete: $V5_GEOMETRY_ROOT"
echo "Next: run scripts/run_visibility.py with your DINOv2 repo/checkpoint."
