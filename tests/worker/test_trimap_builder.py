from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parents[2] / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from worker import trimap_builder  # noqa: E402


class TrimapBuilderTests(unittest.TestCase):
    def test_trimap_has_core_unknown_and_support(self) -> None:
        prior = _rect_mask(10, 10, 3, 3, 4, 4)

        result = trimap_builder.build_trimap_from_prior(prior, b"\xff" * 100, b"\xff" * 100, 10, 10)

        self.assertGreater(result.core_area, 0)
        self.assertGreater(result.unknown_area, 0)
        self.assertGreater(result.support_area, result.prior_area)
        self.assertEqual(result.trimap_mask[5 * 10 + 5], 255)
        self.assertEqual(result.trimap_mask[2 * 10 + 5], 128)
        self.assertEqual(result.trimap_mask[0], 0)

    def test_support_is_clipped_to_region_and_alpha(self) -> None:
        prior = _rect_mask(8, 8, 2, 2, 3, 3)
        region = bytearray(b"\xff" * 64)
        alpha = bytearray(b"\xff" * 64)
        for y in range(8):
            region[y * 8 + 0] = 0
            alpha[y * 8 + 7] = 0

        result = trimap_builder.build_trimap_from_prior(prior, bytes(region), bytes(alpha), 8, 8)

        self.assertTrue(all(result.support_mask[y * 8 + 0] == 0 for y in range(8)))
        self.assertTrue(all(result.support_mask[y * 8 + 7] == 0 for y in range(8)))

    def test_tiny_prior_keeps_non_empty_core(self) -> None:
        prior = bytearray(25)
        prior[2 * 5 + 2] = 255

        result = trimap_builder.build_trimap_from_prior(
            bytes(prior), b"\xff" * 25, b"\xff" * 25, 5, 5
        )

        self.assertEqual(result.prior_area, 1)
        self.assertEqual(result.core_area, 1)
        self.assertEqual(result.core_mask[2 * 5 + 2], 255)

    def test_rejects_mismatched_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "mask sizes do not match"):
            trimap_builder.build_trimap_from_prior(b"\xff", b"\xff", b"\xff", 2, 2)


def _rect_mask(width: int, height: int, x0: int, y0: int, w: int, h: int) -> bytes:
    mask = bytearray(width * height)
    for y in range(y0, y0 + h):
        for x in range(x0, x0 + w):
            mask[y * width + x] = 255
    return bytes(mask)


if __name__ == "__main__":
    unittest.main()
