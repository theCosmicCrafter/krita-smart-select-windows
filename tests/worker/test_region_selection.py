from __future__ import annotations

import sys
import tempfile
import types
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parents[2] / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from worker import region_selection, smart_select_worker  # noqa: E402


class RegionSelectionTests(unittest.TestCase):
    def test_region_mask_bbox_and_prompt_points_are_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            image_path = temp / "snapshot.png"
            image_path.write_bytes(b"placeholder")
            alpha_path = temp / "alpha.npy"
            region_path = temp / "region.npy"
            smart_select_worker.write_npy(alpha_path, "|u1", (8, 8), b"\xff" * 64)
            region = bytearray(64)
            for y in range(2, 6):
                for x in range(1, 5):
                    region[y * 8 + x] = 255
            smart_select_worker.write_npy(region_path, "|u1", (8, 8), bytes(region))

            with self._patched_image_modules(8, 8):
                loaded = region_selection.load_region_selection_input(
                    image_path,
                    alpha_path,
                    region_path,
                    8,
                    8,
                    [1, 2, 4, 4],
                    smart_select_worker.read_npy_uint8_2d,
                )

        self.assertEqual(loaded.region_bbox, (1, 2, 4, 4))
        self.assertGreaterEqual(sum(1 for label in loaded.point_labels if label == 1), 1)
        self.assertTrue(all(label in (0, 1) for label in loaded.point_labels))
        self.assertGreaterEqual(sum(1 for label in loaded.point_labels if label == 0), 1)

        negative_points_inside_lasso = 0
        for point, label in zip(loaded.point_coords, loaded.point_labels, strict=False):
            if label != 0:
                continue
            x = int(round(point[0]))
            y = int(round(point[1]))
            if 0 <= x < 8 and 0 <= y < 8 and loaded.crop_region_mask[y * 8 + x] > 0:
                negative_points_inside_lasso += 1
        self.assertEqual(negative_points_inside_lasso, 0)

        for point, label in zip(loaded.point_coords, loaded.point_labels, strict=False):
            if label != 1:
                continue
            x = int(round(point[0]))
            y = int(round(point[1]))
            self.assertGreater(loaded.crop_region_mask[y * 8 + x], 0)

    def test_empty_region_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            image_path = temp / "snapshot.png"
            image_path.write_bytes(b"placeholder")
            alpha_path = temp / "alpha.npy"
            region_path = temp / "region.npy"
            smart_select_worker.write_npy(alpha_path, "|u1", (4, 4), b"\xff" * 16)
            smart_select_worker.write_npy(region_path, "|u1", (4, 4), b"\x00" * 16)

            with (
                self._patched_image_modules(4, 4),
                self.assertRaises(region_selection.RegionSelectionError),
            ):
                region_selection.load_region_selection_input(
                    image_path,
                    alpha_path,
                    region_path,
                    4,
                    4,
                    [0, 0, 4, 4],
                    smart_select_worker.read_npy_uint8_2d,
                )

    @staticmethod
    @contextmanager
    def _patched_image_modules(width: int, height: int) -> Iterator[None]:
        original_import_module = region_selection.importlib.import_module

        class FakeImage:
            size = (width, height)

            def convert(self, _mode: str) -> FakeImage:
                return self

            def crop(self, _box: tuple[int, int, int, int]) -> FakeImage:
                return self

        def fake_import_module(name: str) -> object:
            if name == "PIL.Image":
                return types.SimpleNamespace(open=lambda _path: FakeImage())
            if name == "numpy":
                return types.SimpleNamespace(array=lambda value, copy=True: value)
            return original_import_module(name)

        region_selection.importlib.import_module = fake_import_module
        try:
            yield
        finally:
            region_selection.importlib.import_module = original_import_module


if __name__ == "__main__":
    unittest.main()
