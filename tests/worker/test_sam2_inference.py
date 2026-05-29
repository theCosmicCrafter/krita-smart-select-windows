from __future__ import annotations

import contextlib
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any

from worker.model_registry import ModelSpec

from worker import sam2_inference


class _InferenceMode:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


class _Cuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeGenerator:
    instances = 0
    last_params: dict[str, object] | None = None

    def __init__(self, _model: object, **params: object) -> None:
        _FakeGenerator.instances += 1
        self.params = params
        _FakeGenerator.last_params = params

    def generate(self, _image_array: object) -> list[dict[str, object]]:
        return [{"segmentation": [[True]], "bbox": [0, 0, 1, 1], "area": 1}]


class _FakePredictor:
    instances = 0
    set_image_calls = 0
    predict_calls: list[dict[str, object]] = []

    def __init__(self, _model: object, **_params: object) -> None:
        _FakePredictor.instances += 1

    def set_image(self, _image_array: object) -> None:
        _FakePredictor.set_image_calls += 1

    def predict(self, **kwargs: object) -> tuple[list[list[list[bool]]], list[float], object]:
        if kwargs.get("multimask_output") is not True:
            raise RuntimeError("multimask_output must be true")
        _FakePredictor.predict_calls.append(kwargs)
        return ([[[True]]], [0.9], object())


class Sam2InferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        sam2_inference._GENERATOR_CACHE.clear()
        sam2_inference._PREDICTOR_CACHE.clear()
        _FakeGenerator.instances = 0
        _FakeGenerator.last_params = None
        _FakePredictor.instances = 0
        _FakePredictor.set_image_calls = 0
        _FakePredictor.predict_calls = []

    def test_generate_masks_uses_cuda_when_available_and_caches_generator(self) -> None:
        with self._patched_imports(cuda_available=True), tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.pt"
            checkpoint.write_bytes(b"checkpoint")

            first = sam2_inference.generate_masks(self._spec(), checkpoint, object())
            second = sam2_inference.generate_masks(self._spec(), checkpoint, object())

        self.assertEqual(first.device, "cuda")
        self.assertEqual(second.device, "cuda")
        self.assertEqual(_FakeGenerator.instances, 1)

    def test_quality_first_generator_parameters_are_used(self) -> None:
        with self._patched_imports(cuda_available=True), tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.pt"
            checkpoint.write_bytes(b"checkpoint")

            sam2_inference.generate_masks(self._spec(), checkpoint, object())

        self.assertIsNotNone(_FakeGenerator.last_params)
        assert _FakeGenerator.last_params is not None
        self.assertEqual(_FakeGenerator.last_params["points_per_side"], 64)
        self.assertEqual(_FakeGenerator.last_params["points_per_batch"], 128)
        self.assertEqual(_FakeGenerator.last_params["crop_n_layers"], 1)
        self.assertEqual(_FakeGenerator.last_params["crop_n_points_downscale_factor"], 1)
        self.assertEqual(_FakeGenerator.last_params["stability_score_thresh"], 0.95)
        self.assertEqual(_FakeGenerator.last_params["min_mask_region_area"], 64)
        self.assertEqual(_FakeGenerator.last_params["use_m2m"], True)

    def test_generate_masks_uses_cpu_when_cuda_is_unavailable(self) -> None:
        with self._patched_imports(cuda_available=False), tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.pt"
            checkpoint.write_bytes(b"checkpoint")

            result = sam2_inference.generate_masks(self._spec(), checkpoint, object())

        self.assertEqual(result.device, "cpu")

    def test_cuda_oom_maps_to_inference_out_of_memory(self) -> None:
        class OomGenerator(_FakeGenerator):
            def generate(self, _image_array: object) -> list[dict[str, object]]:
                raise RuntimeError("CUDA out of memory")

        with (
            self._patched_imports(cuda_available=True, generator_class=OomGenerator),
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            checkpoint = Path(temp_dir) / "model.pt"
            checkpoint.write_bytes(b"checkpoint")

            with self.assertRaises(sam2_inference.InferenceOutOfMemory):
                sam2_inference.generate_masks(self._spec(), checkpoint, object())

    def test_predict_selection_uses_predictor_and_caches_it(self) -> None:
        with self._patched_imports(cuda_available=True), tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.pt"
            checkpoint.write_bytes(b"checkpoint")

            first = sam2_inference.predict_selection(
                self._spec(), checkpoint, object(), (0.0, 0.0, 1.0, 1.0), ((0.0, 0.0),), (1,)
            )
            second = sam2_inference.predict_selection(
                self._spec(), checkpoint, object(), (0.0, 0.0, 1.0, 1.0), ((0.0, 0.0),), (1,)
            )

        self.assertEqual(first.device, "cuda")
        self.assertEqual(second.device, "cuda")
        self.assertEqual(_FakePredictor.instances, 1)
        self.assertEqual(_FakePredictor.set_image_calls, 2)
        self.assertEqual(first.scores, [0.9])

    def test_predict_selection_adds_recall_variants_when_negative_points_exist(self) -> None:
        with self._patched_imports(cuda_available=True), tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.pt"
            checkpoint.write_bytes(b"checkpoint")

            result = sam2_inference.predict_selection(
                self._spec(),
                checkpoint,
                object(),
                (0.0, 0.0, 10.0, 10.0),
                ((4.0, 4.0), (0.0, 0.0)),
                (1, 0),
            )

        self.assertEqual(len(_FakePredictor.predict_calls), 3)
        self.assertEqual(len(result.scores), 3)
        self.assertAlmostEqual(result.scores[0], 0.9)
        self.assertAlmostEqual(result.scores[1], 0.828)
        self.assertAlmostEqual(result.scores[2], 0.756)
        self.assertIsNone(_FakePredictor.predict_calls[2]["point_coords"])
        self.assertIsNone(_FakePredictor.predict_calls[2]["point_labels"])

    @staticmethod
    def _spec() -> ModelSpec:
        return ModelSpec(
            id="sam2.1_hiera_large",
            title="SAM2.1 Hiera Large",
            family="sam2.1",
            checkpoint_url=None,
            checkpoint_filename="sam2.1_hiera_large.pt",
            config_name="configs/sam2.1/sam2.1_hiera_l.yaml",
            expected_bytes=1,
            etag=None,
            recommended_ram_gb=32,
            recommended_vram_gb=12,
            profile="quality",
            requires_download=True,
        )

    @contextlib.contextmanager
    def _patched_imports(
        self,
        *,
        cuda_available: bool,
        generator_class: type[_FakeGenerator] = _FakeGenerator,
    ) -> Any:
        original_import_module = sam2_inference.importlib.import_module

        def fake_import_module(name: str) -> object:
            if name == "torch":
                return types.SimpleNamespace(
                    cuda=_Cuda(cuda_available),
                    inference_mode=lambda: _InferenceMode(),
                    __version__="2.5.1",
                )
            if name == "sam2.build_sam":
                return types.SimpleNamespace(build_sam2=lambda *_args, **_kwargs: object())
            if name == "sam2.automatic_mask_generator":
                return types.SimpleNamespace(SAM2AutomaticMaskGenerator=generator_class)
            if name == "sam2.sam2_image_predictor":
                return types.SimpleNamespace(SAM2ImagePredictor=_FakePredictor)
            if name == "numpy":
                return types.SimpleNamespace(
                    array=lambda value, dtype=None: value,
                    float32="float32",
                    int32="int32",
                )
            return original_import_module(name)

        sam2_inference.importlib.import_module = fake_import_module
        try:
            yield
        finally:
            sam2_inference.importlib.import_module = original_import_module


if __name__ == "__main__":
    unittest.main()
