from __future__ import annotations

import unittest

from worker import segment_map_builder


class SegmentMapBuilderTests(unittest.TestCase):
    def test_alpha_mask_excludes_transparent_pixels(self) -> None:
        mask = [[True, True], [True, True]]

        result = segment_map_builder.build_segment_map_from_masks(
            [{"segmentation": mask, "bbox": [0, 0, 2, 2], "area": 4, "predicted_iou": 1.0}],
            canvas_width=8,
            canvas_height=8,
            inference_width=2,
            inference_height=2,
            alpha_mask=b"\xff\x00\xff\xff",
        )

        self.assertEqual(len(result.metadata), 1)
        self.assertEqual(result.metadata[0]["area"], 48)
        self.assertEqual(result.segment_map[0], 1)
        self.assertEqual(result.segment_map[4], 0)

    def test_overlap_uses_score_then_area_priority(self) -> None:
        broad_mask = [[True for _x in range(10)] for _y in range(10)]
        small_mask = [[False for _x in range(10)] for _y in range(10)]
        for y in range(2, 8):
            for x in range(2, 8):
                small_mask[y][x] = True

        result = segment_map_builder.build_segment_map_from_masks(
            [
                {
                    "segmentation": broad_mask,
                    "bbox": [0, 0, 10, 10],
                    "area": 100,
                    "predicted_iou": 0.70,
                    "stability_score": 0.90,
                },
                {
                    "segmentation": small_mask,
                    "bbox": [2, 2, 6, 6],
                    "area": 36,
                    "predicted_iou": 0.95,
                    "stability_score": 0.90,
                },
            ],
            canvas_width=10,
            canvas_height=10,
            inference_width=10,
            inference_height=10,
            alpha_mask=b"\xff" * 100,
        )

        self.assertEqual(len(result.metadata), 2)
        self.assertEqual(result.segment_map[3 * 10 + 3], 1)
        self.assertEqual(result.segment_map[0], 2)
        self.assertEqual(result.metadata[0]["bbox"], [2, 2, 6, 6])

    def test_tiny_masks_are_dropped(self) -> None:
        mask = [[False for _x in range(8)] for _y in range(8)]
        mask[0][0] = True

        result = segment_map_builder.build_segment_map_from_masks(
            [{"segmentation": mask, "bbox": [0, 0, 1, 1], "area": 1}],
            canvas_width=8,
            canvas_height=8,
            inference_width=8,
            inference_height=8,
            alpha_mask=b"\xff" * 64,
        )

        self.assertEqual(result.metadata, [])
        self.assertTrue(all(value == 0 for value in result.segment_map))


if __name__ == "__main__":
    unittest.main()
