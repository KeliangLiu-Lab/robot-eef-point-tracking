#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from dino_visibility_filter import interpolate_scores, morphology_visibility
from upgrade_geometry_v5 import (
    LEGACY_AUX_OFFSET,
    VISUAL_FLANGE_FACE_OFFSET,
    geometry_from_v4_points,
)


class V5GeometryTest(unittest.TestCase):
    def calibration(self) -> tuple[np.ndarray, np.ndarray]:
        intrinsic = np.asarray(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
        )
        return intrinsic, np.eye(4)

    def test_fast_upgrade_interpolates_visible_flange_in_3d(self) -> None:
        flange = np.asarray([[[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]], np.float32)
        legacy = flange.copy()
        legacy[..., 2] += LEGACY_AUX_OFFSET[2]
        arrays = {
            "flange_footprint": flange,
            "grasp_center_footprint": legacy,
            "operation_mask": np.asarray([True]),
            "operation_range": np.asarray([0, 0], np.int32),
        }
        intrinsic, extrinsic = self.calibration()
        geometry = geometry_from_v4_points(
            arrays, intrinsic, extrinsic, np.eye(4), 100, 80
        )
        np.testing.assert_allclose(
            geometry["track_footprint"] - geometry["flange_footprint"],
            np.broadcast_to(
                VISUAL_FLANGE_FACE_OFFSET, geometry["flange_footprint"].shape
            ),
            atol=1e-7,
        )
        self.assertEqual(
            str(geometry["track_xy_geom"].shape), str((1, 2, 2))
        )
        self.assertNotIn("tcp_xy", geometry)
        self.assertNotIn("grasp_center_xy", geometry)

    def test_right_base_override_reexpresses_saved_left_base_points(self) -> None:
        original_b = np.eye(4)
        original_b[1, 3] = -0.6
        replacement_b = np.eye(4)
        replacement_b[1, 3] = -0.7
        local_left = np.asarray([[0.0, 0.0, 1.0]], np.float32)
        local_right = np.asarray([[0.0, 0.0, 1.0]], np.float32)
        right_in_old_left = local_right.copy()
        right_in_old_left[:, 1] -= 0.6
        flange = np.stack((local_left, right_in_old_left), axis=1)
        legacy = flange.copy()
        legacy[..., 2] += LEGACY_AUX_OFFSET[2]
        arrays = {
            "flange_left_base": flange,
            "grasp_center_left_base": legacy,
            "right_base_to_left_base": original_b,
            "operation_mask": np.asarray([True]),
            "operation_range": np.asarray([0, 0], np.int32),
        }
        intrinsic, extrinsic = self.calibration()
        geometry = geometry_from_v4_points(
            arrays, intrinsic, extrinsic, replacement_b, 100, 80
        )
        np.testing.assert_allclose(
            geometry["flange_footprint"][0, 1], [0.0, -0.7, 1.0], atol=1e-7
        )


class VisibilityMorphologyTest(unittest.TestCase):
    def test_interpolation_is_segment_local(self) -> None:
        sample_indices = np.asarray([0, 5, 10, 15], np.int32)
        sampled = np.asarray(
            [[0.8, np.nan], [np.nan, np.nan], [0.2, np.nan], [np.nan, np.nan]],
            np.float32,
        )
        fov = np.zeros((18, 2), dtype=bool)
        fov[0:3, 0] = True
        fov[8:13, 0] = True
        fov[16:18, 0] = True  # no sample in this short re-entry run
        scores = interpolate_scores(18, sample_indices, sampled, fov)
        np.testing.assert_allclose(scores[0:3, 0], 0.8)
        np.testing.assert_allclose(scores[8:13, 0], 0.2)
        self.assertTrue(np.isnan(scores[3:8, 0]).all())
        self.assertTrue(np.isnan(scores[16:18, 0]).all())

    def test_short_gaps_fill_but_fov_gaps_do_not(self) -> None:
        visible = np.zeros((30, 2), dtype=bool)
        fov = np.ones_like(visible)
        visible[2:10, 0] = True
        visible[13:25, 0] = True
        fov[17:19, 0] = False
        output = morphology_visibility(
            visible, fov, max_false_gap=4, min_true_run=3
        )
        self.assertTrue(output[10:13, 0].all())
        self.assertFalse(output[17:19, 0].any())

    def test_isolated_positive_burst_is_removed(self) -> None:
        visible = np.zeros((20, 2), dtype=bool)
        fov = np.ones_like(visible)
        visible[5:7, 1] = True
        output = morphology_visibility(
            visible, fov, max_false_gap=2, min_true_run=3
        )
        self.assertFalse(output[:, 1].any())


if __name__ == "__main__":
    unittest.main()
