from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from typing import Any

WORKER_DIR = Path(__file__).resolve().parents[2] / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from worker.alpha_model_registry import get_model_spec  # noqa: E402

from worker import birefnet_inference, runtime_env  # noqa: E402


class BiRefNetInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        birefnet_inference._MODEL_CACHE.clear()

    def test_cuda_available_chooses_cuda_and_uses_trust_remote_code(self) -> None:
        with self._patched_runtime(cuda_available=True) as fake:
            result = birefnet_inference.predict_alpha(
                get_model_spec("birefnet_hr_matting"),
                Path("/tmp/model"),
                self._runtime_ready(),
                object(),
            )

        self.assertEqual(result.device, "cuda")
        self.assertEqual(fake.from_pretrained_calls[0]["path"], "/tmp/model")
        self.assertTrue(fake.from_pretrained_calls[0]["trust_remote_code"])
        self.assertEqual(fake.resize_sizes, [(2048, 2048)])

    def test_cuda_unavailable_chooses_cpu(self) -> None:
        with self._patched_runtime(cuda_available=False):
            result = birefnet_inference.predict_alpha(
                get_model_spec("birefnet_hr_matting"),
                Path("/tmp/model"),
                self._runtime_ready(),
                object(),
            )

        self.assertEqual(result.device, "cpu")

    def test_cuda_oom_maps_to_user_safe_error(self) -> None:
        with (
            self._patched_runtime(cuda_available=True, raise_oom=True),
            self.assertRaises(birefnet_inference.InferenceOutOfMemory),
        ):
            birefnet_inference.predict_alpha(
                get_model_spec("birefnet_hr_matting"),
                Path("/tmp/model"),
                self._runtime_ready(),
                object(),
            )

    def test_model_cache_is_keyed_by_model_snapshot_and_device(self) -> None:
        with self._patched_runtime(cuda_available=False) as fake:
            spec = get_model_spec("birefnet_hr_matting")
            birefnet_inference.predict_alpha(
                spec, Path("/tmp/model"), self._runtime_ready(), object()
            )
            birefnet_inference.predict_alpha(
                spec, Path("/tmp/model"), self._runtime_ready(), object()
            )

        self.assertEqual(len(fake.from_pretrained_calls), 1)

    def test_output_alpha_is_resized_to_crop_size_and_soft_values_preserved(self) -> None:
        with self._patched_runtime(cuda_available=False):
            result = birefnet_inference.predict_alpha(
                get_model_spec("birefnet_hr_matting"),
                Path("/tmp/model"),
                self._runtime_ready(),
                object(),
            )

        self.assertEqual((result.width, result.height), (3, 2))
        self.assertEqual(result.crop_alpha, bytes([0, 64, 128, 192, 224, 255]))

    @staticmethod
    def _runtime_ready() -> runtime_env.RuntimeReady:
        return runtime_env.RuntimeReady(
            root=Path("/tmp/runtime"),
            venv_path=Path("/tmp/runtime/venv"),
            python_executable=Path("/tmp/runtime/venv/bin/python"),
            site_packages=Path("/tmp/runtime/venv/site-packages"),
            torch_version="2.6.0",
            cuda_available=True,
        )

    def _patched_runtime(self, cuda_available: bool, raise_oom: bool = False) -> _RuntimePatch:
        return _RuntimePatch(cuda_available, raise_oom)


class _RuntimePatch:
    def __init__(self, cuda_available: bool, raise_oom: bool) -> None:
        self.cuda_available = cuda_available
        self.raise_oom = raise_oom
        self.from_pretrained_calls: list[dict[str, Any]] = []
        self.resize_sizes: list[tuple[int, int]] = []
        self._original_import_module = birefnet_inference.importlib.import_module

    def __enter__(self) -> _RuntimePatch:
        birefnet_inference.importlib.import_module = self._fake_import_module
        return self

    def __exit__(self, *_args: object) -> None:
        birefnet_inference.importlib.import_module = self._original_import_module

    def _fake_import_module(self, name: str) -> object:
        if name == "torch":
            return _FakeTorch(self.cuda_available)
        if name == "PIL.Image":
            return _FakeImageModule()
        if name == "numpy":
            return _FakeNumpy()
        if name == "torchvision.transforms":
            return _FakeTransforms(self)
        if name == "transformers":
            patch = self

            class AutoModelForImageSegmentation:
                @staticmethod
                def from_pretrained(path: str, trust_remote_code: bool) -> _FakeModel:
                    patch.from_pretrained_calls.append(
                        {"path": path, "trust_remote_code": trust_remote_code}
                    )
                    return _FakeModel(patch.raise_oom)

            return types.SimpleNamespace(
                AutoModelForImageSegmentation=AutoModelForImageSegmentation
            )
        return self._original_import_module(name)


class _FakeTorch:
    __version__ = "2.6.0"

    def __init__(self, cuda_available: bool) -> None:
        self.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)

    @staticmethod
    def set_float32_matmul_precision(_mode: str) -> None:
        return None

    @staticmethod
    def inference_mode() -> _NoopContext:
        return _NoopContext()


class _NoopContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


class _FakeModel:
    def __init__(self, raise_oom: bool) -> None:
        self.raise_oom = raise_oom

    def to(self, _device: str) -> _FakeModel:
        return self

    def eval(self) -> _FakeModel:
        return self

    def half(self) -> _FakeModel:
        return self

    def __call__(self, _tensor: object) -> list[_FakePrediction]:
        if self.raise_oom:
            raise RuntimeError("CUDA out of memory")
        return [_FakePrediction()]


class _FakePrediction:
    def sigmoid(self) -> _FakePrediction:
        return self

    def detach(self) -> _FakePrediction:
        return self

    def cpu(self) -> _FakePrediction:
        return self

    def __getitem__(self, _index: int) -> _FakePrediction:
        return self

    def squeeze(self) -> _FakePrediction:
        return self

    def float(self) -> _FakePrediction:
        return self


class _FakeImage:
    size = (3, 2)

    def convert(self, _mode: str) -> _FakeImage:
        return self

    def resize(self, _size: tuple[int, int], _resampling: object) -> _FakeImage:
        return self


class _FakeImageModule:
    Resampling = types.SimpleNamespace(BILINEAR="bilinear")

    @staticmethod
    def fromarray(_array: object) -> _FakeImage:
        return _FakeImage()


class _FakeTransforms:
    def __init__(self, patch: _RuntimePatch) -> None:
        self.patch = patch

    def Compose(self, steps: list[object]) -> object:  # noqa: N802
        def apply(_image: object) -> _FakeTensor:
            return _FakeTensor()

        return apply

    def Resize(self, size: tuple[int, int]) -> object:  # noqa: N802
        self.patch.resize_sizes.append(size)
        return object()

    @staticmethod
    def ToTensor() -> object:  # noqa: N802
        return object()

    @staticmethod
    def Normalize(_mean: list[float], _std: list[float]) -> object:  # noqa: N802
        return object()

    @staticmethod
    def ToPILImage() -> object:  # noqa: N802
        return lambda _prediction: _FakeImage()


class _FakeTensor:
    def unsqueeze(self, _dim: int) -> _FakeTensor:
        return self

    def to(self, _device: str) -> _FakeTensor:
        return self

    def half(self) -> _FakeTensor:
        return self


class _FakeNumpy:
    uint8 = "uint8"

    @staticmethod
    def asarray(_image: object, dtype: object) -> _FakeArray:
        return _FakeArray()


class _FakeArray:
    def reshape(self, _size: int) -> _FakeArray:
        return self

    @staticmethod
    def tobytes() -> bytes:
        return bytes([0, 64, 128, 192, 224, 255])


if __name__ == "__main__":
    unittest.main()
