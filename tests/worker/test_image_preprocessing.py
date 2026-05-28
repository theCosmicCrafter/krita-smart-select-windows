from __future__ import annotations

import unittest

from worker.image_preprocessing import MAX_INFERENCE_LONG_SIDE, inference_size


class ImagePreprocessingTests(unittest.TestCase):
    def test_small_image_keeps_original_size(self) -> None:
        self.assertEqual(inference_size(1024, 768), (1024, 768))

    def test_large_landscape_image_caps_long_side(self) -> None:
        width, height = inference_size(4096, 2048)

        self.assertEqual(width, MAX_INFERENCE_LONG_SIDE)
        self.assertEqual(height, 1024)

    def test_large_portrait_image_caps_long_side(self) -> None:
        width, height = inference_size(1000, 4000)

        self.assertEqual(width, 512)
        self.assertEqual(height, MAX_INFERENCE_LONG_SIDE)


if __name__ == "__main__":
    unittest.main()
