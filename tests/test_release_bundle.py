from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agx_calibration_registry import CalibrationRegistry
from backfill_agilex7000_shared_geometry import load_override as load_shared_override
from upgrade_geometry_v5 import load_override as load_v5_overrides


class ReleaseBundleTest(unittest.TestCase):
    def test_bundled_calibration_registry_loads_reviewed_profiles(self) -> None:
        registry = CalibrationRegistry()
        self.assertEqual(
            registry.select("cup", 0).calibration_id,
            "mobile_aloha_front_episode2_canary",
        )
        self.assertEqual(
            registry.select("ordered", 0).calibration_id,
            "ordered_color_blocks_front_dino_control_debiased_v1",
        )
        self.assertEqual(
            registry.select("color", 0).calibration_id,
            "color_blocks_front_period1_episode0_v1",
        )

    def test_portable_episode_overrides_load_and_match_expected_scope(self) -> None:
        expected = {
            1952: "agilex7000-ep1952-e217-v1",
            2829: "agilex7000-ep2829-source44-eb-v1",
        }
        loaded = {}
        for path in sorted((ROOT / "configs/overrides").glob("agilex7000_ep*.json")):
            loaded.update(load_v5_overrides(path))
        self.assertEqual(len(loaded), 2)
        for episode, version in expected.items():
            payload = loaded[("agilex7000_manip_short30_rot6d_rowmajor_h32_v4_prompt_reviewed", episode)]
            self.assertEqual(payload["override_version"], version)
            self.assertTrue(payload["portable_release_calibration"])
            self.assertEqual(payload["_matrix"].shape, (4, 4))
            self.assertEqual(payload["_replacement_intrinsic"].shape, (3, 3))
        self.assertIsNone(
            loaded[("agilex7000_manip_short30_rot6d_rowmajor_h32_v4_prompt_reviewed", 1952)]["_right_to_left"]
        )
        self.assertEqual(
            loaded[("agilex7000_manip_short30_rot6d_rowmajor_h32_v4_prompt_reviewed", 2829)]["_right_to_left"].shape,
            (4, 4),
        )

    def test_shared_calibration_override_evidence_is_portable_and_hashed(self) -> None:
        path = ROOT / "configs/overrides/agilex7000_hanging_shared_right_base_invariant.json"
        payload = load_shared_override(
            path, "agilex7000_manip_short30_rot6d_rowmajor_h32_v4_prompt_reviewed"
        )
        self.assertEqual(payload["episode_range"], {"start_inclusive": 11910, "stop_exclusive": 11970})
        self.assertEqual(payload["_intrinsic"].shape, (3, 3))
        self.assertEqual(payload["_camera_to_left"].shape, (4, 4))
        self.assertEqual(payload["_right_to_left"].shape, (4, 4))

    def test_evidence_hashes_match_pinned_release_values(self) -> None:
        expected = {
            "physical_flange_evidence.json": "896c846c39ea5e44c5d92b8b7408693086a7a28ec1df85e40301eb86a5f4d798",
            "ep1952_calibration_score_evidence.json": "b4dc2cafb557df7c59e42579eabc1658ad4b81a406469d2973349bd2eb05b678",
            "ep2829_calibration_override_evidence.json": "fd8d8126f1d97e6420bd4109a3b153f04720750a0c0c04dba5c097da38865213",
            "shared_right_base_initializer_evidence.json": "0d726eff64bddf58a1b4dfe3e2ab86b45112aabbd6e9e418f369c8bf65531f27",
        }
        for name, digest in expected.items():
            path = ROOT / "assets" / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
