#!/usr/bin/env python3

import unittest

import numpy as np

from batch_project_agilex7000_geometry import (
    CalibrationError,
    GeometryError,
    canonical_array_signature,
    make_manifest_record,
    points_in_fov,
    prepare_projected_point_output,
    project_points,
    reconstruct_geometry,
    rot6d_rows_to_matrix,
    transform_points,
    validate_intrinsic,
    validate_rigid_transform,
)


class AgileX7000GeometryTest(unittest.TestCase):
    def test_row_major_rot6d_and_local_z_offset(self) -> None:
        # +90 degrees about Y: flange-local +Z points along base +X.
        rotation = np.array(
            [[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]], dtype=np.float64
        )
        matrix = rot6d_rows_to_matrix(rotation)
        expected = np.array(
            [[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]]
        )
        np.testing.assert_allclose(matrix, expected, atol=1e-12)
        offset = np.einsum("tij,j->ti", matrix, np.array([0.0, 0.0, 0.13503]))
        np.testing.assert_allclose(offset, [[0.13503, 0.0, 0.0]], atol=1e-12)

    def test_rot6d_rejects_degenerate_rows(self) -> None:
        with self.assertRaises(GeometryError):
            rot6d_rows_to_matrix(np.zeros((1, 6)))
        with self.assertRaises(GeometryError):
            rot6d_rows_to_matrix(np.array([[1.0, 0.0, 0.0, 2.0, 0.0, 0.0]]))

    def test_transform_points_uses_right_base_to_left_base(self) -> None:
        right_to_left = np.eye(4)
        right_to_left[:3, 3] = [1.0, -2.0, 0.5]
        point_right = np.array([[0.2, 0.3, 0.4]])
        np.testing.assert_allclose(
            transform_points(right_to_left, point_right), [[1.2, -1.7, 0.9]]
        )

    def test_pinhole_projection(self) -> None:
        intrinsic = np.array(
            [[100.0, 0.0, 10.0], [0.0, 200.0, 20.0], [0.0, 0.0, 1.0]]
        )
        xy, depth = project_points(
            np.array([[1.0, 2.0, 2.0]]), intrinsic, np.eye(4)
        )
        np.testing.assert_allclose(xy, [[60.0, 220.0]])
        np.testing.assert_allclose(depth, [2.0])

    def test_camera_to_base_is_inverted(self) -> None:
        intrinsic = np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
        )
        camera_to_base = np.eye(4)
        camera_to_base[0, 3] = 1.0
        base_to_camera = np.linalg.inv(camera_to_base)
        xy, depth = project_points(
            np.array([[1.0, 0.0, 2.0]]), intrinsic, base_to_camera
        )
        np.testing.assert_allclose(xy, [[50.0, 40.0]])
        np.testing.assert_allclose(depth, [2.0])

    def test_cached_inverse_can_be_injected(self) -> None:
        states = np.zeros((1, 20), dtype=np.float64)
        states[0, 0:3] = [1.0, 0.0, 2.0]
        states[0, 3:9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        states[0, 10:13] = [1.0, 0.0, 2.0]
        states[0, 13:19] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        intrinsic = np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
        )
        camera_to_base = np.eye(4)
        camera_to_base[0, 3] = 1.0
        geometry = reconstruct_geometry(
            states,
            intrinsic,
            camera_to_base,
            np.eye(4),
            100,
            80,
            left_base_to_camera=np.linalg.inv(camera_to_base),
        )
        np.testing.assert_allclose(geometry["flange_xy_geom"], [[[50.0, 40.0]] * 2])

    def test_reconstruct_dual_arm_chain_and_fov(self) -> None:
        states = np.zeros((1, 20), dtype=np.float64)
        identity_rot6d_rows = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        states[0, 0:3] = [0.0, 0.0, 1.0]
        states[0, 3:9] = identity_rot6d_rows
        states[0, 10:13] = [0.0, 0.0, 1.0]
        states[0, 13:19] = identity_rot6d_rows
        intrinsic = np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
        )
        right_to_left = np.eye(4)
        right_to_left[0, 3] = 0.1
        geometry = reconstruct_geometry(
            states, intrinsic, np.eye(4), right_to_left, 100, 80
        )
        np.testing.assert_allclose(
            geometry["flange_left_base"],
            [[[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]],
        )
        np.testing.assert_allclose(
            geometry["grasp_center_left_base"],
            [[[0.0, 0.0, 1.13503], [0.1, 0.0, 1.13503]]],
        )
        self.assertTrue(geometry["flange_geometric_in_fov"].all())
        self.assertTrue(geometry["grasp_center_geometric_in_fov"].all())

    def test_fov_requires_positive_depth_and_half_open_bounds(self) -> None:
        xy = np.array([[0.0, 0.0], [99.999, 79.999], [100.0, 20.0], [5.0, 5.0]])
        depth = np.array([1.0, 1.0, 1.0, -1.0])
        np.testing.assert_array_equal(
            points_in_fov(xy, depth, 100, 80), [True, True, False, False]
        )

    def test_stored_fov_is_computed_after_float32_quantization(self) -> None:
        xy64 = np.array([[[1279.99999, 300.0]]], dtype=np.float64)
        depth64 = np.array([[1.0]], dtype=np.float64)
        raw, depth, in_fov, masked = prepare_projected_point_output(
            xy64, depth64, 1280, 720
        )
        self.assertEqual(raw.dtype, np.float32)
        self.assertEqual(depth.dtype, np.float32)
        self.assertEqual(float(raw[0, 0, 0]), 1280.0)
        self.assertFalse(in_fov[0, 0])
        self.assertTrue(np.isnan(masked[0, 0]).all())

    def test_matrix_validation_thresholds(self) -> None:
        qa = validate_rigid_transform(np.eye(4), "identity")
        self.assertEqual(qa.determinant, 1.0)
        bad_bottom = np.eye(4)
        bad_bottom[3, 0] = 1e-3
        with self.assertRaises(CalibrationError):
            validate_rigid_transform(bad_bottom, "bad_bottom")
        reflection = np.eye(4)
        reflection[0, 0] = -1.0
        with self.assertRaises(CalibrationError):
            validate_rigid_transform(reflection, "reflection")
        non_orthogonal = np.eye(4)
        non_orthogonal[0, 0] = 1.01
        with self.assertRaises(CalibrationError):
            validate_rigid_transform(non_orthogonal, "scaled")

    def test_intrinsic_validation(self) -> None:
        intrinsic = np.array(
            [[900.0, 0.0, 640.0], [0.0, 901.0, 360.0], [0.0, 0.0, 1.0]]
        )
        qa = validate_intrinsic(intrinsic)
        self.assertEqual(qa["fx"], 900.0)
        intrinsic[2, 0] = 1e-3
        with self.assertRaises(CalibrationError):
            validate_intrinsic(intrinsic)

    def test_signature_is_shape_and_value_sensitive(self) -> None:
        matrix = np.eye(4)
        first = canonical_array_signature("E", matrix)
        self.assertEqual(first, canonical_array_signature("E", matrix.copy()))
        changed = matrix.copy()
        changed[0, 3] = 1e-12
        self.assertNotEqual(first, canonical_array_signature("E", changed))
        self.assertNotEqual(first, canonical_array_signature("B", matrix))

    def test_manifest_exposes_overall_and_per_matrix_calibration_status(self) -> None:
        base = {
            "dataset": "dataset",
            "episode_index": 7,
            "video": "/video.mp4",
            "geometry_npz": "/geometry.npz",
            "geometry_report": "/geometry.report.json",
            "source_parquet": "/source.parquet",
            "status": "success",
            "calibration_status": {"K": "valid", "E": "valid", "B": "valid"},
        }
        valid = make_manifest_record(base)
        self.assertEqual(valid["calibration_status"], "valid")
        self.assertEqual(valid["calibration_matrices"], base["calibration_status"])
        missing_report = dict(base)
        missing_report["calibration_status"] = {
            "K": "valid",
            "E": "missing",
            "B": "valid",
        }
        missing = make_manifest_record(missing_report)
        self.assertEqual(missing["calibration_status"], "missing")

    def test_manifest_preserves_shared_override_provenance(self) -> None:
        report = {
            "dataset": "dataset",
            "episode_index": 11910,
            "video": "/video.mp4",
            "geometry_npz": "/geometry.npz",
            "geometry_report": "/geometry.report.json",
            "source_parquet": "/source.parquet",
            "status": "success",
            "calibration_status": {"K": "valid", "E": "valid", "B": "valid"},
            "calibration_method": "shared_right_base_invariant",
            "calibration_evidence": "/evidence.json",
            "calibration_override": {"version": "override-v1"},
            "calibration": {"signatures": {"combined": "signature"}},
        }
        record = make_manifest_record(report)
        self.assertEqual(record["calibration_method"], "shared_right_base_invariant")
        self.assertEqual(record["calibration_evidence"], "/evidence.json")
        self.assertEqual(record["calibration_override_version"], "override-v1")
        self.assertEqual(record["calibration_signature"], "signature")


if __name__ == "__main__":
    unittest.main()
