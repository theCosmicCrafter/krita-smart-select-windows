from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parents[2] / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from worker import mam_refiner, refiner_registry, trimap_builder  # noqa: E402


class MamRefinerTests(unittest.TestCase):
    def test_guidance_uses_sam2_prior_when_sam_prior_is_fallback(self) -> None:
        refinement_input = self._input(prior_fallback_used=True, prior_region_coverage=0.25)

        self.assertEqual(mam_refiner._guidance_source(refinement_input), "sam2_prior")

    def test_guidance_uses_sam2_prior_when_sam_prior_covers_most_region(self) -> None:
        refinement_input = self._input(prior_fallback_used=False, prior_region_coverage=0.95)

        self.assertEqual(mam_refiner._guidance_source(refinement_input), "sam2_prior")

    def test_guidance_uses_sam2_prior_for_specific_prior(self) -> None:
        refinement_input = self._input(prior_fallback_used=False, prior_region_coverage=0.50)

        self.assertEqual(mam_refiner._guidance_source(refinement_input), "sam2_prior")

    def test_refinement_gate_accepts_small_changes(self) -> None:
        prior = bytes([255] * 100)
        candidate = bytes([255] * 96 + [0] * 4)

        result = mam_refiner._refinement_gate(candidate, prior)

        self.assertTrue(result.accepted)
        self.assertIsNone(result.reason)

    def test_refinement_gate_rejects_excessive_area_growth(self) -> None:
        prior = bytes([255] * 100 + [0] * 50)
        candidate = bytes([255] * 125 + [0] * 25)

        result = mam_refiner._refinement_gate(candidate, prior)

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "area_too_large")

    def test_refinement_gate_rejects_excessive_area_loss(self) -> None:
        prior = bytes([255] * 100)
        candidate = bytes([255] * 60 + [0] * 40)

        result = mam_refiner._refinement_gate(candidate, prior)

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "area_too_small")

    def test_runtime_payload_marks_rejected_mam(self) -> None:
        spec = refiner_registry.get_refiner_spec("mam_sam_vith")
        result = mam_refiner.MamRefinementResult(
            crop_alpha=b"\xff",
            device="cuda",
            torch_version="2.5.1",
            cuda_available=True,
            duration_ms=1,
            fallback_used=True,
            rejected=True,
            rejection_reason="area_too_large",
        )

        payload = mam_refiner.runtime_result_payload(result, spec)

        self.assertEqual(payload["cache_state"], "rejected_mam")
        self.assertEqual(payload["rejection_reason"], "area_too_large")

    @staticmethod
    def _input(
        *, prior_fallback_used: bool, prior_region_coverage: float
    ) -> mam_refiner.MamRefinementInput:
        return mam_refiner.MamRefinementInput(
            crop_rgb=object(),
            crop_alpha_mask=b"\xff",
            crop_region_mask=b"\xff",
            crop_sam_prior_mask=b"\xff",
            trimap=trimap_builder.TrimapBuildResult(
                trimap_mask=b"\xff",
                core_mask=b"\xff",
                unknown_mask=b"\x00",
                support_mask=b"\xff",
                core_erode_radius=1,
                support_dilate_radius=2,
                prior_area=1,
                core_area=1,
                unknown_area=0,
                support_area=1,
            ),
            crop_bbox=(0, 0, 1, 1),
            prompt_box_xyxy=(0.0, 0.0, 1.0, 1.0),
            canvas_width=1,
            canvas_height=1,
            prior_fallback_used=prior_fallback_used,
            prior_region_coverage=prior_region_coverage,
        )


if __name__ == "__main__":
    unittest.main()
