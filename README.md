# Calibrated Robot End-Effector Point Tracking

Calibration-driven 2D end-effector labels for the main
manipulation camera of five AgileX/Piper LeRobot datasets.

**This repository packages the calibrated labeling pipeline.** It does not
include datasets, videos, robot URDF files, DINOv2 source code, or model
weights. DINOv2 only gates visibility; it never moves the geometrically
projected point.

## Reference Run

The source run covered five datasets, 13,622 episodes, and 12,315,388 frames.
The final validation completed all 13,622 episodes with zero missing or
invalid outputs. The DINOv2 stage inferred 12,215 episodes and reused 1,407
previously verified outputs after checking their identities. These totals
describe the audited reference run; generated labels and video assets are not
bundled here.

The audited scope is:

| Dataset | Episodes | Main view |
|---|---:|---|
| Mobile Cup Tray | 534 | `observation.images.cam_manip_high` |
| Mobile White Box | 477 | `observation.images.cam_manip_high` |
| Mobile Color Blocks | 274 | `observation.images.cam_manip_high` |
| AgileX7000 manipulation | 12,150 | `observation.images.camera_front` |
| Ordered Color Blocks | 187 | `observation.images.cam_manip_high` |

Navigation and wrist cameras are excluded. The three mobile datasets use their
reviewed operation-phase boundaries.

## Why Coordinates Are Stable

The pipeline separates coordinates from visibility:

```text
Cartesian robot state + physical point definition + calibrated K/E/B
                                  |
                                  v
                    deterministic pixel coordinate
                                  |
main-view image -> DINOv2 appearance gate -> visible/uncertain mask
```

1. Coordinates come from the logged Cartesian J6/link6 state and a
   separately named point on the robot. They are not free-running visual
   tracker outputs.
2. Calibrations are selected by source episode provenance, not guessed from
   renumbered final episode IDs. Intrinsic K, camera extrinsic E, and right to
   left base transform B are validated and recorded.
3. The 6D rotations are decoded with the dataset's row-major convention.
   Right-arm points are transformed through the calibrated base relationship
   before projection.
4. Nonpositive depth, out-of-frame points, navigation intervals, and rejected
   operation phases are explicitly masked. Coordinates are never clamped to
   image edges or propagated from the last visible frame.
5. DINOv2 determines visibility at the exact projected feature token. It does
   not search broadly for a nearby match and cannot modify the point position.
6. Every episode output has a report and content identity. Resume checks
   validate the state/video/geometry inputs and output hashes.

### Point Contract

- `eef_xy_geom` / `j6_origin_xy_geom`: canonical logged J6/link6 origin.
- `track_xy_geom`: the selected visible flange/front-plate candidate at
  link6-local `[0, 0, 0.056] m`. URDF mesh topology supports a connected annular
  +Z face with a central bore and bolt holes. Confidence is **medium** because
  the URDF does not declare a named CAD mounting datum.
- `track_xy`: `track_xy_geom` only where DINO confirms visibility; otherwise
  NaN.
- `legacy_aux_xy_geom`: retained only for diagnostic compatibility. It is not
  asserted to be TCP, contact point, or flange.

The canonical J6 point and visible front-face candidate are intentionally
separate. No unique TCP/contact point is claimed. The included
`assets/physical_flange_evidence.json` records the topology audit. The source
AGILEX URDF and its referenced meshes are external inputs and are not bundled.
Internal `algorithm_version` and `schema_version` values in NPZ files and
reports are compatibility identifiers for the audited data, not package
release names.

### Reviewed Calibration Corrections

- AgileX7000 final episode 1952: retain source-216 K/B and replace only camera
  E with the validated source-217 camera transform.
- AgileX7000 final episode 2829: retain K and replace E and B together with
  source-44 values. Replacing E alone causes approximately 100 px right-arm
  error in the visible interval.
- AgileX7000 final episodes 11910-11969: use a shared right-base-invariant
  calibration. The included initializer evidence reports leave-one-group-out
  median projection errors of about 12.5, 31.0, and 15.5 px across three
  calibration groups. These episodes should be considered lower-confidence
  than the ordinary source-calibrated episodes and reviewed for the intended
  use case.

The per-episode override matrices, expected original signatures, and supporting
evidence summaries are included under `configs/overrides/` and `assets/`. The
pipeline checks that the input episode contains the expected original
calibration before applying an override. It never switches calibration per
frame.

## Scope and Limitations

- The reviewed calibration and point definition apply only to the five listed
  AgileX/Piper datasets. Other robots and cameras require their own point
  definition, calibration review, and validation.
- Fifty-five AgileX7000 episodes were flagged by visibility-distribution
  auditing for review. A flagged episode is not automatically invalid, but
  users should inspect these cases for their application.
- DINO visibility is appearance evidence, not a physical occlusion sensor.
- The visual front-face point is medium-confidence and does not replace the
  robot's canonical J6 state.
- The published statistics describe the exact audited source run. If data,
  videos, calibration metadata, preprocessing, or DINO prototypes change, run
  the complete validation again and do not assume the same accuracy.

## Repository Layout

```text
src/
  batch_project_agx_geometry.py          AGX mobile/ordered calibrated geometry
  batch_project_agilex7000_geometry.py   AgileX7000 calibrated geometry
  backfill_agilex7000_shared_geometry.py validated 60-episode shared-E backfill
  project_flange_geometry.py             final J6/flange semantic geometry stage
  dino_visibility_filter.py             DINOv2 feature scorer and filters
  dino_eef_visibility_batch.py          sharded, resumable visibility stage
  validate_labels.py                     full output and provenance validation
  render_preview.py                     H.264/yuv420p review video renderer
configs/
  calibration_template/                 reviewed camera K/E calibration inputs
  overrides/                            three narrow, evidence-backed overrides
  dino_prototypes.json                  positive and negative reference points
assets/                                     portable calibration/point evidence
tests/                                      geometry and visibility behavior tests
scripts/                                   end-to-end stage launchers
```

The LaTeX pseudocode is in `docs/ALGORITHMS.tex`, and an English method
summary is in `docs/METHOD_DESCRIPTION.md`. The algorithms cover calibrated
coordinates, DINOv2 visibility, and end-to-end dataset generation.

## Requirements

- Linux, Python 3.10-3.12
- NumPy, PyArrow, PyYAML, OpenCV, and PyTorch
- CUDA-capable PyTorch for the DINOv2 visibility stage
- `ffmpeg` and `ffprobe` for H.264 review previews
- The local DINOv2 repository and a compatible ViT-B/14 checkpoint
- The five final LeRobot datasets and their original videos/calibration fields

Install CPU-side dependencies in a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the PyTorch build matching your CUDA driver using the official PyTorch
installation instructions. Clone or otherwise provide a compatible local
DINOv2 repository separately. This package deliberately does not vendor
third-party model code or weights.

The visibility code loads the `dinov2_vitb14` architecture from
https://github.com/facebookresearch/dinov2 using a PyTorch state dictionary
(or a checkpoint with a `model` entry). The checkpoint must match that
architecture exactly. Before inference, verify that every entry in
`configs/dino_prototypes.json` points to an available source video and a
reviewed flange location in original-image pixels. Re-review these prototypes
when changing cameras or robot appearance.

## Input Data Contract

`EEF_DATA_ROOT` must contain these dataset directories:

```text
agx_cup_tray_mobile_rot6d_vlash_se2_h48_phase_split_v3_action_current_v1/
agx_move_white_box_mobile_rot6d_vlash_se2_h48_phase_split_v2_action_current_v1/
agx_color_blocks_mobile_rot6d_vlash_se2_h48_phase_split_v3_action_current_v1/
agilex7000_manip_short30_rot6d_rowmajor_h32_v4_prompt_reviewed/
agx_ordered_color_blocks_rot6d_vlash_se2_h32_v2_action_current_v1/
```

Inputs follow the LeRobot layout and must include `meta/info.json`,
`meta/episodes.jsonl`, per-episode parquet, and main-view video files. AGX
calibration fields must be available in the AGX episode parquet files and
resolve to the calibration policy represented by `configs/calibration_template`.
AgileX7000 inputs must include its source episode map and source calibration
columns. The 60-episode shared-E scope must match the included override's
episode range and source dataset contract.

Do not use a dataset with a different episode ordering, calibration source
mapping, video mapping, state convention, or camera resolution without
reviewing and adapting the calibration policy first.

## Full Pipeline

Run from this repository directory. Outputs are placed under `outputs/` by
default and are excluded by `.gitignore`.

```bash
export EEF_DATA_ROOT=/path/to/lerobot
export PYTHON=python
export WORKERS=24
bash scripts/run_geometry.sh
```

The geometry runner performs, in order:

1. AGX mobile/ordered projection using the included calibration registry.
2. AgileX7000 projection from its per-episode calibration metadata.
3. The validated shared calibration backfill for episodes 11910-11969.
4. Flange-face reprojection and the two episode-level
   calibration corrections.

Review the geometry-only projections before starting visibility inference.
The final geometry root defaults to:

```text
outputs/flange_geometry
```

Run the DINOv2 visibility stage on the selected GPU IDs:

```bash
python scripts/run_visibility.py \
  --data-root "$EEF_DATA_ROOT" \
  --manifest outputs/flange_geometry/geometry_manifest.jsonl \
  --geometry-root outputs/flange_geometry \
  --output-root outputs/eef_tracks \
  --dinov2-repo /path/to/dinov2 \
  --checkpoint /path/to/dinov2_vitb14_pretrain.pth \
  --devices 0 \
  --workers-per-device 1 \
  --batch-size 16
```

This example uses one GPU. For more throughput, list several GPU IDs with
`--devices`, then increase `--workers-per-device` and `--batch-size` only
as GPU memory permits. Prototype videos must be available under the data root
on every worker.

Each DINO process handles a deterministic manifest shard. It resumes only
outputs whose geometry, target video, prototype videos, code, checkpoint, and
fusion configuration identities match. Logs and JSONL events are written
under `outputs/eef_tracks/logs/`.

Validate the complete result:

```bash
EEF_FLANGE_GEOMETRY_ROOT=outputs/flange_geometry \
EEF_FINAL_ROOT=outputs/eef_tracks \
bash scripts/run_validation.sh
```

The validator checks the complete manifest, schema, point contract, frame
counts, operation/FOV/visibility masks, output digests, video fingerprints,
geometry identities, and calibration override provenance. A successful exit
with `--require-complete` is required before treating a run as complete.
The supplied validation launcher enforces the exact five-dataset episode
counts listed above; for a trial or subset, call `src/validate_labels.py`
directly on its manifest without `--require-complete`:

```bash
python src/validate_labels.py \
  --manifest outputs/flange_geometry/geometry_manifest.partial.jsonl \
  --output-root outputs/eef_tracks \
  --report outputs/eef_tracks/subset_validation.json \
  --workers 4 --verify-sha256
```

A partial geometry run writes `geometry_manifest.partial.jsonl` unless an
explicit manifest path was requested.

## Review Preview

Render an H.264/yuv420p video that is compatible with VS Code and common media
players:

```bash
bash scripts/render_preview.sh \
  /path/to/main_view.mp4 \
  outputs/eef_tracks/DATASET/chunk-000/episode_000000.npz \
  /tmp/episode_000000_preview.mp4 \
  2
```

Colored dots/trails show `track_xy` (the visually confirmed candidate). White
crosses show the canonical J6 origin. If the track is not visible, its trail
breaks instead of following the last location.

## Reading Labels

```python
import numpy as np

with np.load("episode_000000.npz", allow_pickle=False) as data:
    xy = data["track_xy_geom"].astype(np.float32)  # [T,2,2], [left/right, x/y]
    in_operation_fov = data["operation_track_geometric_in_fov"].astype(bool)
    visible = data["track_visible"].astype(bool)
    width, height = data["image_size"].astype(int)

# Coordinate regression when the point is geometrically inside the operation view.
coordinate_mask = in_operation_fov

# More conservative loss using only DINO-confirmed visible points.
visible_coordinate_mask = in_operation_fov & visible
```

Use `track_xy_geom` for target coordinates and an explicit mask for the loss.
Do not pass `track_xy` directly into a regression loss because it contains NaNs
for invisible points. The `quality` array is 0 for invalid, 1 for
visible, and 2 for in-frame but uncertain or occluded points. If training
images are resized, cropped, or assembled into
a mosaic, apply exactly the same transform to the coordinates before
normalizing them.

## Tests

Run CPU unit tests without datasets or model weights:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The full-data validation additionally requires the user-supplied datasets and
videos. A DINO smoke test requires a CUDA-capable PyTorch install, DINOv2
source, checkpoint, and the prototype source videos.

## Calibration Evidence

The `assets/` directory contains machine-path-scrubbed evidence for:

- the Piper link6 axial mesh-topology audit and the medium-confidence 56 mm
  visible face candidate;
- the E216/E217 episode-1952 calibration comparison;
- the coupled E44/B44 episode-2829 override comparison and temporal check;
- the shared-right-base-invariant initializer for the 60 AgileX7000 episodes.

The numeric matrices in the overrides and calibration templates are the
production values. Evidence images, videos, user data, and original calibration
source files are not included.

## Licensing and Third-Party Components

This repository is licensed under the MIT License. DINOv2 source and weights,
PyTorch, OpenCV, PyArrow, and all robot datasets remain subject to their own
licenses and terms. They are not redistributed here.
