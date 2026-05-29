from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
WORKER_DIR = ROOT_DIR / "worker"
WORKER_PATH = ROOT_DIR / "worker" / "smart_select_worker.py"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from refiner_cache import RefinerCacheStatus  # noqa: E402


def load_worker_module() -> Any:
    spec = importlib.util.spec_from_file_location("smart_select_worker", WORKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load worker module from {WORKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_npy_header(path: Path) -> tuple[str, tuple[int, ...], bytes]:
    data = path.read_bytes()
    if not data.startswith(b"\x93NUMPY\x01\x00"):
        raise AssertionError("expected NumPy v1 .npy file")

    header_length = struct.unpack("<H", data[8:10])[0]
    header = data[10 : 10 + header_length].decode("latin1")
    descr = header.split("'descr': '", 1)[1].split("'", 1)[0]
    shape_text = header.split("'shape': (", 1)[1].split(")", 1)[0]
    shape = tuple(int(value.strip()) for value in shape_text.split(",") if value.strip())
    payload = data[10 + header_length :]
    return descr, shape, payload


def uint16_values(payload: bytes) -> tuple[int, ...]:
    return struct.unpack("<" + "H" * (len(payload) // 2), payload)


class SmartSelectWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.worker = load_worker_module()
        self.worker.cancelled_jobs.clear()

    def test_fake_segments_returns_three_hit_testable_segments(self) -> None:
        segment_map, metadata = self.worker.fake_segments(64, 48)

        self.assertEqual(len(segment_map), 64 * 48)
        self.assertEqual([item["id"] for item in metadata], [1, 2, 3])
        self.assertTrue(all(item["area"] > 0 for item in metadata))
        self.assertEqual(segment_map[4 * 64 + 5], 1)
        self.assertEqual(segment_map[24 * 64 + 32], 2)
        self.assertEqual(segment_map[32 * 64 + 43], 3)

    def test_prepare_segments_emits_progress_result_and_uint16_npy(self) -> None:
        output = io.StringIO()
        cleanup_dir: Path | None = None
        try:
            with contextlib.redirect_stdout(output):
                self.worker.handle_prepare_segments(
                    7,
                    {
                        "image_path": "",
                        "model_id": "mock",
                        "fingerprint": "mock-no-snapshot-v1",
                        "canvas_width": 32,
                        "canvas_height": 24,
                    },
                )

            messages = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(
                [message.get("stage") for message in messages[:4]],
                [
                    "resolving_model",
                    "reading_image",
                    "generating_segments",
                    "building_hover_index",
                ],
            )

            result = messages[-1]["result"]
            segment_map_path = Path(result["segment_map_path"])
            segments_path = Path(result["segments_path"])
            cleanup_dir = segment_map_path.parent
            self.assertTrue(segment_map_path.exists())
            self.assertTrue(segments_path.exists())
            self.assertEqual(len(result["metadata"]), 3)
            self.assertEqual(result["model"]["id"], "mock")

            descr, shape, payload = parse_npy_header(segment_map_path)
            self.assertEqual(descr, "<u2")
            self.assertEqual(shape, (24, 32))
            self.assertEqual(len(payload), 24 * 32 * 2)
        finally:
            if cleanup_dir is not None:
                shutil.rmtree(cleanup_dir, ignore_errors=True)

    def test_prepare_selection_mock_returns_uint8_selection_mask(self) -> None:
        output = io.StringIO()
        cleanup_dir: Path | None = None
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            image_path = temp / "snapshot.png"
            image_path.write_bytes(b"placeholder")
            alpha_path = temp / "alpha.npy"
            region_path = temp / "region.npy"
            self.worker.write_npy(alpha_path, "|u1", (8, 8), b"\xff" * 64)
            region = bytearray(64)
            for y in range(2, 6):
                for x in range(2, 6):
                    region[y * 8 + x] = 255
            self.worker.write_npy(region_path, "|u1", (8, 8), bytes(region))

            original_load_region_selection_input = (
                self.worker.region_selection.load_region_selection_input
            )

            def fake_load_region_selection_input(
                _image_path: str | Path,
                _alpha_mask_path: str | Path,
                _region_mask_path: str | Path,
                _canvas_width: int,
                _canvas_height: int,
                _requested_region_bbox: list[int],
                _alpha_reader: Any,
            ) -> Any:
                return self.worker.region_selection.RegionSelectionInput(
                    crop_image_array=object(),
                    crop_alpha_mask=b"\xff" * 64,
                    crop_region_mask=bytes(region),
                    crop_bbox=(0, 0, 8, 8),
                    region_bbox=(2, 2, 4, 4),
                    prompt_box_xyxy=(2.0, 2.0, 6.0, 6.0),
                    point_coords=((4.0, 4.0),),
                    point_labels=(1,),
                    canvas_width=8,
                    canvas_height=8,
                )

            self.worker.region_selection.load_region_selection_input = (
                fake_load_region_selection_input
            )
            try:
                with contextlib.redirect_stdout(output):
                    self.worker.handle_prepare_selection(
                        71,
                        {
                            "image_path": str(image_path),
                            "alpha_mask_path": str(alpha_path),
                            "region_mask_path": str(region_path),
                            "region_bbox": [2, 2, 4, 4],
                            "model_id": "mock",
                            "fingerprint": "fingerprint",
                            "canvas_width": 8,
                            "canvas_height": 8,
                        },
                    )

                messages = [json.loads(line) for line in output.getvalue().splitlines()]
                result = messages[-1]["result"]
                selection_mask_path = Path(result["selection_mask_path"])
                cleanup_dir = selection_mask_path.parent
                self.assertTrue(selection_mask_path.exists())
                descr, shape, payload = parse_npy_header(selection_mask_path)
                self.assertEqual(descr, "|u1")
                self.assertEqual(shape, (8, 8))
                self.assertEqual(payload[2 * 8 + 2], 255)
                self.assertEqual(payload[0], 0)
            finally:
                self.worker.region_selection.load_region_selection_input = (
                    original_load_region_selection_input
                )
                if cleanup_dir is not None:
                    shutil.rmtree(cleanup_dir, ignore_errors=True)

    def test_prepare_selection_real_model_emits_refiner_stages_and_soft_mask(self) -> None:
        output = io.StringIO()
        cleanup_dir: Path | None = None
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            image_path = temp / "snapshot.png"
            image_path.write_bytes(b"placeholder")
            alpha_path = temp / "alpha.npy"
            region_path = temp / "region.npy"
            self.worker.write_npy(alpha_path, "|u1", (4, 4), b"\xff" * 16)
            self.worker.write_npy(region_path, "|u1", (4, 4), b"\xff" * 16)
            model_cache_root = temp / "models"
            model_dir = model_cache_root / "sam2.1_hiera_large"
            model_dir.mkdir(parents=True)
            (model_dir / "sam2.1_hiera_large.pt").write_bytes(b"verified")
            spec = self.worker.ModelSpec(
                id="sam2.1_hiera_large",
                title="SAM2.1 Hiera Large",
                family="sam2.1",
                checkpoint_url="https://example.invalid/model.pt",
                checkpoint_filename="sam2.1_hiera_large.pt",
                config_name="configs/sam2.1/sam2.1_hiera_l.yaml",
                expected_bytes=8,
                etag='"etag"',
                recommended_ram_gb=32,
                recommended_vram_gb=12,
                profile="quality",
                requires_download=True,
            )
            refiner_spec = self.worker.get_refiner_spec("mam_sam_vith")
            selection_input = self.worker.region_selection.RegionSelectionInput(
                crop_image_array=object(),
                crop_alpha_mask=b"\xff" * 16,
                crop_region_mask=b"\xff" * 16,
                crop_bbox=(0, 0, 4, 4),
                region_bbox=(0, 0, 4, 4),
                prompt_box_xyxy=(0.0, 0.0, 4.0, 4.0),
                point_coords=((2.0, 2.0),),
                point_labels=(1,),
                canvas_width=4,
                canvas_height=4,
            )

            original_get_model_spec = self.worker.get_model_spec
            original_load_region = self.worker.region_selection.load_region_selection_input
            original_ensure_runtime = self.worker.runtime_env.ensure_runtime
            original_ensure_mam_source = self.worker.runtime_env.ensure_mam_source
            original_predict_selection = self.worker.sam2_inference.predict_selection
            original_refiner_cache = self.worker.RefinerCache
            original_refine = self.worker.mam_refiner.refine_with_mam
            self.worker.get_model_spec = lambda _model_id: spec
            self.worker.region_selection.load_region_selection_input = (
                lambda *_args, **_kwargs: selection_input
            )

            def fake_ensure_runtime(_root: Path, progress: Any, _context: Any = None) -> Any:
                progress("checking_runtime", None, {})
                return self.worker.runtime_env.RuntimeReady(
                    root=temp / "runtime",
                    venv_path=temp / "runtime" / "venv",
                    python_executable=temp / "runtime" / "venv" / "bin" / "python",
                    site_packages=temp / "runtime" / "venv" / "site-packages",
                    torch_version="2.5.1",
                    cuda_available=True,
                    cuda_home="/usr/local/cuda-12.4",
                    sam2_cuda_extension=True,
                )

            class FakeRefinerCache:
                def __init__(self, _root: Path) -> None:
                    pass

                def ensure_downloaded(self, _spec: Any, progress: Any, _cancelled: Any) -> Any:
                    progress("downloading_refiner", 1.0, {})
                    return RefinerCacheStatus(
                        refiner_id=refiner_spec.id,
                        state="verified",
                        sam_checkpoint_path=temp / "sam.pth",
                        mam_checkpoint_path=temp / "mam.pth",
                        sam_actual_bytes=refiner_spec.sam_expected_bytes,
                        mam_actual_bytes=refiner_spec.mam_expected_bytes,
                    )

            self.worker.runtime_env.ensure_runtime = fake_ensure_runtime
            self.worker.runtime_env.ensure_mam_source = lambda *_args, **_kwargs: (
                self.worker.runtime_env.MamSourceReady(temp / "source", refiner_spec.source_ref)
            )
            self.worker.sam2_inference.predict_selection = lambda *_args, **_kwargs: (
                self.worker.sam2_inference.PromptedSelectionResult(
                    masks=[[[True for _x in range(4)] for _y in range(4)]],
                    scores=[0.9],
                    device="cuda",
                    torch_version="2.5.1",
                    cuda_available=True,
                )
            )
            self.worker.RefinerCache = FakeRefinerCache
            self.worker.mam_refiner.refine_with_mam = lambda *_args, **_kwargs: (
                self.worker.mam_refiner.MamRefinementResult(
                    crop_alpha=bytes([0, 64, 128, 255] * 4),
                    device="cuda",
                    torch_version="2.5.1",
                    cuda_available=True,
                    duration_ms=1,
                )
            )
            try:
                with contextlib.redirect_stdout(output):
                    self.worker.handle_prepare_selection(
                        72,
                        {
                            "image_path": str(image_path),
                            "alpha_mask_path": str(alpha_path),
                            "region_mask_path": str(region_path),
                            "region_bbox": [0, 0, 4, 4],
                            "model_cache_dir": str(model_cache_root),
                            "runtime_cache_dir": str(temp / "runtime"),
                            "refiner_cache_dir": str(temp / "refiners"),
                            "model_id": "sam2.1_hiera_large",
                            "refiner_id": "mam_sam_vith",
                            "fingerprint": "fingerprint",
                            "canvas_width": 4,
                            "canvas_height": 4,
                        },
                    )
            finally:
                self.worker.get_model_spec = original_get_model_spec
                self.worker.region_selection.load_region_selection_input = original_load_region
                self.worker.runtime_env.ensure_runtime = original_ensure_runtime
                self.worker.runtime_env.ensure_mam_source = original_ensure_mam_source
                self.worker.sam2_inference.predict_selection = original_predict_selection
                self.worker.RefinerCache = original_refiner_cache
                self.worker.mam_refiner.refine_with_mam = original_refine

            messages = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertIn("resolving_refiner", [message.get("stage") for message in messages])
            self.assertIn("running_mam", [message.get("stage") for message in messages])
            self.assertIn("building_alpha_mask", [message.get("stage") for message in messages])
            result = messages[-1]["result"]
            cleanup_dir = Path(result["selection_mask_path"]).parent
            descr, shape, payload = parse_npy_header(Path(result["selection_mask_path"]))
            self.assertEqual(descr, "|u1")
            self.assertEqual(shape, (4, 4))
            self.assertEqual(payload[1], 64)
            self.assertEqual(payload[2], 128)
            self.assertEqual(result["refiner"]["id"], "mam_sam_vith")
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    def test_prepare_selection_real_model_without_refiner_cache_dir_returns_error(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            image_path = temp / "snapshot.png"
            image_path.write_bytes(b"placeholder")
            alpha_path = temp / "alpha.npy"
            region_path = temp / "region.npy"
            self.worker.write_npy(alpha_path, "|u1", (4, 4), b"\xff" * 16)
            self.worker.write_npy(region_path, "|u1", (4, 4), b"\xff" * 16)
            model_cache_root = temp / "models"
            model_dir = model_cache_root / "sam2.1_hiera_large"
            model_dir.mkdir(parents=True)
            (model_dir / "sam2.1_hiera_large.pt").write_bytes(b"verified")
            spec = self.worker.ModelSpec(
                id="sam2.1_hiera_large",
                title="SAM2.1 Hiera Large",
                family="sam2.1",
                checkpoint_url="https://example.invalid/model.pt",
                checkpoint_filename="sam2.1_hiera_large.pt",
                config_name="configs/sam2.1/sam2.1_hiera_l.yaml",
                expected_bytes=8,
                etag='"etag"',
                recommended_ram_gb=32,
                recommended_vram_gb=12,
                profile="quality",
                requires_download=True,
            )
            selection_input = self.worker.region_selection.RegionSelectionInput(
                crop_image_array=object(),
                crop_alpha_mask=b"\xff" * 16,
                crop_region_mask=b"\xff" * 16,
                crop_bbox=(0, 0, 4, 4),
                region_bbox=(0, 0, 4, 4),
                prompt_box_xyxy=(0.0, 0.0, 4.0, 4.0),
                point_coords=((2.0, 2.0),),
                point_labels=(1,),
                canvas_width=4,
                canvas_height=4,
            )
            original_get_model_spec = self.worker.get_model_spec
            original_load_region = self.worker.region_selection.load_region_selection_input
            original_ensure_runtime = self.worker.runtime_env.ensure_runtime
            original_predict_selection = self.worker.sam2_inference.predict_selection
            self.worker.get_model_spec = lambda _model_id: spec
            self.worker.region_selection.load_region_selection_input = (
                lambda *_args, **_kwargs: selection_input
            )
            self.worker.runtime_env.ensure_runtime = lambda *_args, **_kwargs: (
                self.worker.runtime_env.RuntimeReady(
                    root=temp / "runtime",
                    venv_path=temp / "runtime" / "venv",
                    python_executable=temp / "runtime" / "venv" / "bin" / "python",
                    site_packages=temp / "runtime" / "venv" / "site-packages",
                    torch_version="2.5.1",
                    cuda_available=True,
                    cuda_home="/usr/local/cuda-12.4",
                    sam2_cuda_extension=True,
                )
            )
            self.worker.sam2_inference.predict_selection = lambda *_args, **_kwargs: (
                self.worker.sam2_inference.PromptedSelectionResult(
                    masks=[[[True for _x in range(4)] for _y in range(4)]],
                    scores=[0.9],
                    device="cuda",
                    torch_version="2.5.1",
                    cuda_available=True,
                )
            )
            try:
                with contextlib.redirect_stdout(output):
                    self.worker.handle_prepare_selection(
                        73,
                        {
                            "image_path": str(image_path),
                            "alpha_mask_path": str(alpha_path),
                            "region_mask_path": str(region_path),
                            "region_bbox": [0, 0, 4, 4],
                            "model_cache_dir": str(model_cache_root),
                            "runtime_cache_dir": str(temp / "runtime"),
                            "model_id": "sam2.1_hiera_large",
                            "canvas_width": 4,
                            "canvas_height": 4,
                        },
                    )
            finally:
                self.worker.get_model_spec = original_get_model_spec
                self.worker.region_selection.load_region_selection_input = original_load_region
                self.worker.runtime_env.ensure_runtime = original_ensure_runtime
                self.worker.sam2_inference.predict_selection = original_predict_selection

            messages = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(messages[-1]["error"]["type"], "RefinerCacheDirMissing")

    def test_crop_mask_prompt_box_uses_nonzero_prior_bounds(self) -> None:
        mask = bytearray(100)
        for y in range(4, 7):
            for x in range(3, 6):
                mask[y * 10 + x] = 255

        box = self.worker.crop_mask_prompt_box(bytes(mask), (0, 0, 10, 10))

        self.assertEqual(box, (1.0, 2.0, 8.0, 9.0))

    def test_prepare_segments_rejects_invalid_canvas_size(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.worker.handle_prepare_segments(8, {"canvas_width": 0, "canvas_height": 24})

        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(
            messages,
            [
                {
                    "id": 8,
                    "error": {
                        "type": "InvalidInput",
                        "message": "Canvas size is invalid",
                    },
                }
            ],
        )

    def test_read_npy_uint8_2d_accepts_valid_alpha_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "alpha.npy"
            self.worker.write_npy(path, "|u1", (2, 3), b"\x00\x01\x02\x03\x04\x05")

            width, height, payload = self.worker.read_npy_uint8_2d(path)

        self.assertEqual((width, height), (3, 2))
        self.assertEqual(payload, b"\x00\x01\x02\x03\x04\x05")

    def test_read_npy_uint8_2d_rejects_wrong_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "alpha.npy"
            self.worker.write_npy(path, "<u2", (2, 3), b"\x00" * 12)

            with self.assertRaisesRegex(ValueError, "Expected uint8 alpha mask"):
                self.worker.read_npy_uint8_2d(path)

    def test_prepare_segments_alpha_mask_excludes_transparent_pixels(self) -> None:
        output = io.StringIO()
        cleanup_dir: Path | None = None
        with tempfile.TemporaryDirectory() as temp_dir:
            alpha_path = Path(temp_dir) / "alpha.npy"
            alpha = bytearray([255] * (32 * 24))
            for y in range(2, 10):
                for x in range(2, 10):
                    alpha[y * 32 + x] = 0
            self.worker.write_npy(alpha_path, "|u1", (24, 32), bytes(alpha))

            try:
                with contextlib.redirect_stdout(output):
                    self.worker.handle_prepare_segments(
                        12,
                        {
                            "image_path": "",
                            "alpha_mask_path": str(alpha_path),
                            "model_id": "mock",
                            "algorithm_version": "mock-worker-snapshot-v1",
                            "fingerprint": "fingerprint",
                            "canvas_width": 32,
                            "canvas_height": 24,
                        },
                    )

                messages = [json.loads(line) for line in output.getvalue().splitlines()]
                result = messages[-1]["result"]
                cleanup_dir = Path(result["segment_map_path"]).parent
                self.assertEqual([item["id"] for item in result["metadata"]], [2, 3])

                descr, shape, payload = parse_npy_header(Path(result["segment_map_path"]))
                self.assertEqual(descr, "<u2")
                self.assertEqual(shape, (24, 32))
                values = uint16_values(payload)
                self.assertEqual(values[2 * 32 + 2], 0)
                self.assertEqual(values[12 * 32 + 16], 2)
            finally:
                if cleanup_dir is not None:
                    shutil.rmtree(cleanup_dir, ignore_errors=True)

    def test_prepare_segments_rejects_alpha_mask_size_mismatch(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            alpha_path = Path(temp_dir) / "alpha.npy"
            self.worker.write_npy(alpha_path, "|u1", (2, 3), b"\xff" * 6)

            with contextlib.redirect_stdout(output):
                self.worker.handle_prepare_segments(
                    13,
                    {
                        "model_id": "mock",
                        "alpha_mask_path": str(alpha_path),
                        "canvas_width": 32,
                        "canvas_height": 24,
                    },
                )

        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(messages[-1]["error"]["type"], "InvalidInput")
        self.assertEqual(
            messages[-1]["error"]["message"], "Alpha mask size does not match canvas size"
        )

    def test_prepare_segments_real_model_with_verified_cache_uses_model_stages(self) -> None:
        output = io.StringIO()
        cleanup_dir: Path | None = None
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir) / "models"
            model_dir = cache_root / "sam2.1_hiera_large"
            model_dir.mkdir(parents=True)
            checkpoint = model_dir / "sam2.1_hiera_large.pt"
            checkpoint.write_bytes(b"verified")
            image_path = Path(temp_dir) / "snapshot.png"
            alpha_path = Path(temp_dir) / "alpha.npy"
            image_path.write_bytes(b"placeholder")
            self.worker.write_npy(alpha_path, "|u1", (24, 32), b"\xff" * (32 * 24))
            spec = self.worker.ModelSpec(
                id="sam2.1_hiera_large",
                title="SAM2.1 Hiera Large",
                family="sam2.1",
                checkpoint_url="https://example.invalid/model.pt",
                checkpoint_filename="sam2.1_hiera_large.pt",
                config_name="configs/sam2.1/sam2.1_hiera_l.yaml",
                expected_bytes=8,
                etag='"etag"',
                recommended_ram_gb=32,
                recommended_vram_gb=12,
                profile="quality",
                requires_download=True,
            )
            original_get_model_spec = self.worker.get_model_spec
            original_ensure_runtime = self.worker.runtime_env.ensure_runtime
            original_load_preprocessed = self.worker.image_preprocessing.load_preprocessed_image
            original_generate_masks = self.worker.sam2_inference.generate_masks
            self.worker.get_model_spec = lambda _model_id: spec

            def fake_ensure_runtime(_root: Path, progress: Any, _context: Any = None) -> Any:
                progress("checking_runtime", None, {})
                progress("validating_runtime", None, {})
                return self.worker.runtime_env.RuntimeReady(
                    root=Path(temp_dir) / "runtime",
                    venv_path=Path(temp_dir) / "runtime" / "venv",
                    python_executable=Path(temp_dir) / "runtime" / "venv" / "bin" / "python",
                    site_packages=Path(temp_dir) / "runtime" / "venv" / "site-packages",
                    torch_version="2.5.1",
                    cuda_available=True,
                    cuda_home="/usr/local/cuda-12.4",
                    sam2_cuda_extension=True,
                )

            mask = [[False for _x in range(32)] for _y in range(24)]
            for y in range(4, 14):
                for x in range(5, 17):
                    mask[y][x] = True

            def fake_load_preprocessed(
                _image_path: str | Path,
                _alpha_mask_path: str | Path,
                _canvas_width: int,
                _canvas_height: int,
                _alpha_reader: Any,
            ) -> Any:
                return self.worker.image_preprocessing.PreprocessedImage(
                    image_array=object(),
                    alpha_mask=b"\xff" * (32 * 24),
                    canvas_width=32,
                    canvas_height=24,
                    inference_width=32,
                    inference_height=24,
                    resized=False,
                )

            def fake_generate_masks(
                _spec: Any, _checkpoint_path: Path, _image_array: object
            ) -> Any:
                return self.worker.sam2_inference.Sam2InferenceResult(
                    masks=[
                        {
                            "segmentation": mask,
                            "bbox": [5, 4, 12, 10],
                            "area": 120,
                            "predicted_iou": 0.98,
                            "stability_score": 0.96,
                        }
                    ],
                    device="cuda",
                    torch_version="2.5.1",
                    cuda_available=True,
                )

            self.worker.runtime_env.ensure_runtime = fake_ensure_runtime
            self.worker.image_preprocessing.load_preprocessed_image = fake_load_preprocessed
            self.worker.sam2_inference.generate_masks = fake_generate_masks
            try:
                with contextlib.redirect_stdout(output):
                    self.worker.handle_prepare_segments(
                        14,
                        {
                            "image_path": str(image_path),
                            "alpha_mask_path": str(alpha_path),
                            "model_cache_dir": str(cache_root),
                            "runtime_cache_dir": str(Path(temp_dir) / "runtime"),
                            "model_id": "sam2.1_hiera_large",
                            "algorithm_version": "sam2.1-auto-mask-quality-v1",
                            "fingerprint": "fingerprint",
                            "canvas_width": 32,
                            "canvas_height": 24,
                        },
                    )
            finally:
                self.worker.get_model_spec = original_get_model_spec
                self.worker.runtime_env.ensure_runtime = original_ensure_runtime
                self.worker.image_preprocessing.load_preprocessed_image = original_load_preprocessed
                self.worker.sam2_inference.generate_masks = original_generate_masks

            messages = [json.loads(line) for line in output.getvalue().splitlines()]
            cleanup_dir = Path(messages[-1]["result"]["segment_map_path"]).parent
            try:
                self.assertEqual(
                    [message.get("stage") for message in messages[:11]],
                    [
                        "resolving_model",
                        "checking_runtime",
                        "validating_runtime",
                        "checking_model_cache",
                        "verifying_model",
                        "reading_image",
                        "preprocessing_image",
                        "loading_model",
                        "running_sam",
                        "building_segment_map",
                        "building_hover_index",
                    ],
                )
                self.assertEqual(messages[-1]["result"]["model"]["id"], "sam2.1_hiera_large")
                self.assertEqual(messages[-1]["result"]["model"]["checkpoint_bytes"], 8)
                self.assertEqual(messages[-1]["result"]["runtime"]["device"], "cuda")
            finally:
                if cleanup_dir is not None:
                    shutil.rmtree(cleanup_dir, ignore_errors=True)

    def test_prepare_segments_real_model_without_runtime_dir_returns_error(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir, contextlib.redirect_stdout(output):
            image_path = Path(temp_dir) / "snapshot.png"
            alpha_path = Path(temp_dir) / "alpha.npy"
            image_path.write_bytes(b"placeholder")
            self.worker.write_npy(alpha_path, "|u1", (24, 32), b"\xff" * (32 * 24))
            self.worker.handle_prepare_segments(
                15,
                {
                    "image_path": str(image_path),
                    "alpha_mask_path": str(alpha_path),
                    "model_id": "sam2.1_hiera_large",
                    "canvas_width": 32,
                    "canvas_height": 24,
                },
            )

        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(messages[-1]["error"]["type"], "RuntimeCacheDirMissing")

    def test_prepare_segments_unknown_model_returns_error(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.worker.handle_prepare_segments(
                16,
                {
                    "model_id": "unknown",
                    "canvas_width": 32,
                    "canvas_height": 24,
                },
            )

        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(messages[-1]["error"]["type"], "UnknownModel")

    def test_handle_request_rejects_non_object_params(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.worker.handle_request({"id": 9, "method": "prepare_segments", "params": []})

        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(messages[0]["error"]["type"], "InvalidInput")

    def test_diagnostics_request_returns_runtime_info(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir, contextlib.redirect_stdout(output):
            self.worker.handle_request(
                {
                    "id": 17,
                    "method": "diagnostics",
                    "params": {"model_cache_dir": temp_dir},
                }
            )

        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(messages[0]["id"], 17)
        self.assertIn("python", messages[0]["result"])
        self.assertIn("models", messages[0]["result"])

    def test_cancel_request_records_cancelled_job(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.worker.handle_request({"id": 10, "method": "cancel", "params": {"job_id": 7}})

        self.assertIn(7, self.worker.cancelled_jobs)
        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(messages[0], {"id": 10, "result": {"cancelled": True}})

    def test_worker_process_accepts_ndjson_request(self) -> None:
        request = {
            "id": 11,
            "method": "prepare_segments",
            "params": {
                "image_path": "",
                "model_id": "mock",
                "fingerprint": "mock-no-snapshot-v1",
                "canvas_width": 16,
                "canvas_height": 12,
            },
        }

        completed = subprocess.run(
            [sys.executable, str(WORKER_PATH)],
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        messages = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual(messages[-1]["id"], 11)
        self.assertIn("result", messages[-1])
        cleanup_dir = Path(messages[-1]["result"]["segment_map_path"]).parent
        shutil.rmtree(cleanup_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
