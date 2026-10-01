#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from agx_calibration_registry import Calibration, CalibrationRegistry, DATASET_SPECS
from batch_project_agx_geometry import (
    ALGORITHM_VERSION,
    WorkItem,
    _operation_start,
    compute_geometry,
    geometry_manifest_rows,
    project_episode,
    rot6d_rows_to_matrix,
)


def calibration_payload(calibration_id: str) -> dict:
    return {
        "calibration_id": calibration_id,
        "status": "CAMERA_CALIBRATION_UNIT_VALIDATED",
        "intrinsic": {"matrix": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]},
        "extrinsic": {
            "direction": "footprint_to_camera",
            "matrix": np.eye(4).tolist(),
        },
    }


def write_yaml(path: Path, calibration_id: str) -> None:
    path.write_text(yaml.safe_dump(calibration_payload(calibration_id)), encoding="utf-8")


def make_calibration(dataset_key: str = "cup") -> Calibration:
    return Calibration(
        dataset_key=dataset_key,
        source_episode_index=7,
        calibration_id="test_calibration",
        calibration_unit="test",
        calibration_path=Path("test.yaml"),
        calibration_sha256="a" * 64,
        intrinsic=np.asarray(calibration_payload("x")["intrinsic"]["matrix"]),
        extrinsic=np.eye(4),
        yaml_status="VALIDATED",
        manifest_status=None,
        selection_mode="fixed",
        provisional=False,
        provisional_reason=None,
        calibration_method=None,
        evidence=None,
    )


def identity_states(frame_count: int) -> np.ndarray:
    states = np.zeros((frame_count, 23), dtype=np.float32)
    # Select local positions so both mounted J6 origins are footprint [0, 0, 1].
    states[:, 3:6] = [-0.23875, -0.3, 0.225]
    states[:, 13:16] = [-0.23875, 0.3, 0.225]
    states[:, 6:12] = [1, 0, 0, 0, 1, 0]
    states[:, 16:22] = [1, 0, 0, 0, 1, 0]
    return states


class CalibrationRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        write_yaml(self.root / "mobile_aloha_front_episode2_canary.yaml", "cup")
        write_yaml(
            self.root / "ordered_color_blocks_front_dino_control_debiased_v1.yaml",
            "ordered_color_blocks_front_dino_control_debiased_v1",
        )
        write_yaml(
            self.root / "color_blocks_front_period2_episode274_v1.yaml", "color_p2"
        )
        write_yaml(self.root / "white.yaml", "white_p2")
        write_yaml(self.root / "color_p1.yaml", "color_p1")
        (self.root / "move_white_box_manifest.jsonl").write_text(
            json.dumps(
                {
                    "calibration_unit": "white_period2",
                    "source_episode_indices": [102, 103],
                    "source_episode_count": 2,
                    "calibration": "calibration/white.yaml",
                    "calibration_id": "white_p2",
                    "status": "VALIDATED",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (self.root / "color_blocks_manifest.jsonl").write_text(
            json.dumps(
                {
                    "calibration_unit": "color_period1",
                    "source_episode_indices": [0, 4],
                    "source_episode_count": 2,
                    "calibration": "calibration/color_p1.yaml",
                    "status": "VALIDATED",
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_source_episode_lookup_and_validated_ordered(self) -> None:
        registry = CalibrationRegistry(self.root)
        self.assertEqual(registry.select("white", 102).calibration_id, "white_p2")
        self.assertEqual(registry.select("color", 4).calibration_id, "color_p1")
        self.assertEqual(registry.select("cup", 9999).calibration_id, "cup")
        ordered = registry.select("ordered", 0)
        self.assertEqual(
            ordered.calibration_id,
            "ordered_color_blocks_front_dino_control_debiased_v1",
        )
        self.assertFalse(ordered.provisional)
        self.assertIsNone(ordered.provisional_reason)
        with self.assertRaisesRegex(KeyError, "absent from the calibration manifest"):
            registry.select("color", 1)

    def test_duplicate_manifest_source_is_rejected(self) -> None:
        path = self.root / "color_blocks_manifest.jsonl"
        row = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "occurs more than once"):
            CalibrationRegistry(self.root)


class GeometryProjectionTest(unittest.TestCase):
    def test_rot6d_is_row_major(self) -> None:
        rotation = rot6d_rows_to_matrix(np.asarray([[0, -1, 0, 1, 0, 0]]))[0]
        np.testing.assert_allclose(
            rotation,
            np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            atol=1e-7,
        )

    def test_geometric_fov_is_independent_from_phase(self) -> None:
        geometry = compute_geometry(
            identity_states(3), make_calibration(), (100, 100), operation_start=1
        )
        self.assertTrue(geometry["geometric_in_fov"].all())
        self.assertTrue(geometry["flange_geometric_in_fov"].all())
        np.testing.assert_array_equal(
            geometry["geometric_in_fov"],
            geometry["grasp_center_geometric_in_fov"],
        )
        self.assertFalse(geometry["operation_mask"][0])
        self.assertTrue(np.isnan(geometry["grasp_center_xy"][0]).all())
        self.assertTrue(np.isfinite(geometry["grasp_center_projected_xy"][0]).all())
        np.testing.assert_array_equal(
            geometry["flange_xy_geom"], geometry["flange_projected_xy"]
        )
        np.testing.assert_array_equal(
            geometry["grasp_center_xy_geom"],
            geometry["grasp_center_projected_xy"],
        )
        np.testing.assert_allclose(
            geometry["grasp_center_footprint"][:, :, 2], 1.13503, atol=1e-6
        )

    def test_phase_contract(self) -> None:
        spec = DATASET_SPECS["cup"]
        self.assertEqual(
            _operation_start(
                spec,
                0,
                10,
                {0: {"status": "ok", "frame_count": 10, "split_frame": 4}},
            ),
            (4, "ok"),
        )
        self.assertEqual(
            _operation_start(
                spec,
                0,
                10,
                {0: {"status": "no_navigation_motion", "frame_count": 10}},
            ),
            (0, "no_navigation_motion"),
        )
        self.assertEqual(
            _operation_start(
                spec,
                0,
                10,
                {0: {"status": "insufficient_terminal_static", "frame_count": 10}},
            ),
            (-1, "insufficient_terminal_static"),
        )


class AtomicOutputTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.parquet = self.root / "episode.parquet"
        states = identity_states(4)
        state_array = pa.FixedSizeListArray.from_arrays(
            pa.array(states.reshape(-1)), list_size=23
        )
        table = pa.table(
            {
                "observation.state": state_array,
                "source_episode_index": pa.array([7] * 4, type=pa.int64()),
            }
        )
        pq.write_table(table, self.parquet)
        self.item = WorkItem(
            dataset_key="cup",
            dataset_name=DATASET_SPECS["cup"].dataset_name,
            dataset_root=str(self.root),
            episode_index=3,
            source_episode_index=7,
            expected_frames=4,
            parquet_path=str(self.parquet),
            video_path=str(self.root / "episode.mp4"),
            image_size=(100, 100),
            operation_start=2,
            phase_status="ok",
            phase_metadata_path="phase.jsonl",
            output_npz=str(self.root / "output.npz"),
            output_report=str(self.root / "output.report.json"),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_atomic_pair_and_resume(self) -> None:
        calibration = make_calibration()

        class Registry:
            def select(self, dataset: str, source_episode_index: int) -> Calibration:
                self.last = (dataset, source_episode_index)
                return calibration

        registry = Registry()
        first = project_episode(self.item, registry)
        self.assertEqual(first["status"], "completed")
        report = json.loads(Path(self.item.output_report).read_text(encoding="utf-8"))
        self.assertEqual(report["algorithm_version"], ALGORITHM_VERSION)
        self.assertEqual(report["phase"]["excluded_navigation_or_unusable_frames"], 2)
        with np.load(self.item.output_npz, allow_pickle=False) as archive:
            self.assertEqual(archive["geometric_in_fov"].shape, (4, 2))
            self.assertTrue(archive["geometric_in_fov"].all())
            self.assertFalse(archive["operation_mask"][0])
        second = project_episode(self.item, registry)
        self.assertEqual(second["status"], "skipped")
        leftovers = [path for path in self.root.iterdir() if path.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_manifest_contract(self) -> None:
        calibration = make_calibration()

        class Registry:
            def select(self, dataset: str, source_episode_index: int) -> Calibration:
                return calibration

        row = geometry_manifest_rows(
            [self.item], Registry(), self.root / "final"
        )[0]
        for key in (
            "dataset",
            "episode_index",
            "video",
            "geometry_npz",
            "geometry_report",
            "calibration_id",
            "calibration_status",
            "output_npz",
        ):
            self.assertIn(key, row)
        self.assertEqual(row["calibration_status"], "validated")


if __name__ == "__main__":
    unittest.main()
