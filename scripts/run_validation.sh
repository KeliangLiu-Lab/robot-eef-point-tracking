#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_ROOT"
PYTHON="${PYTHON:-python}"
GEOMETRY_ROOT="${EEF_V5_GEOMETRY_ROOT:-$PACKAGE_ROOT/outputs/eef_tracks_calibrated_v5_geometry_final}"
FINAL_ROOT="${EEF_FINAL_ROOT:-$PACKAGE_ROOT/outputs/eef_tracks_calibrated_v5}"

"$PYTHON" src/validate_v5.py \
  --manifest "$GEOMETRY_ROOT/geometry_manifest.jsonl" \
  --output-root "$FINAL_ROOT" \
  --report "$FINAL_ROOT/full_validation.json" \
  --workers "${VALIDATION_WORKERS:-16}" \
  --verify-sha256 \
  --require-complete
