from __future__ import annotations

import contextlib
import io
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

    def test_unsupported_gpu_raises_clear_error_without_cpu_fallback(self) -> None:
        stderr = io.StringIO()
        with (
            self._patched_runtime(
                cuda_available=True, capability=(12, 0), arch_list=("sm_50", "sm_90")
            ) as fake,
            contextlib.redirect_stderr(stderr),
            self.assertRaises(birefnet_inference.InferenceUnsupportedGpu) as raised,
        ):
            birefnet_inference.predict_alpha(
                get_model_spec("birefnet_hr_matting"),
                Path("/tmp/model"),
                self._runtime_ready(),
                object(),
            )

        message = str(raised.exception)
        self.assertIn("Fake GPU", message)
        self.assertIn("compute capability 12.0", message)
        self.assertIn("sm_50 sm_90", message)
        self.assertIn("CUDA 12.8", message)
        self.assertEqual(fake.from_pretrained_calls, [])
        log = stderr.getvalue()
        self.assertIn("event.name=alpha_model.cuda.incompatible", log)
        self.assertIn("severity=error", log)
        self.assertIn("device.compute_capability=12.0", log)
        self.assertIn('torch.cuda.arch_list=["sm_50", "sm_90"]', log)
        self.assertIn("error.reason=cuda_arch_not_supported", log)

    def test_blackwell_build_runs_on_blackwell_gpu(self) -> None:
        with self._patched_runtime(
            cuda_available=True, capability=(12, 0), arch_list=("sm_90", "sm_100", "sm_120")
        ):
            result = birefnet_inference.predict_alpha(
                get_model_spec("birefnet_hr_matting"),
                Path("/tmp/model"),
                self._runtime_ready(),
                object(),
            )

        self.assertEqual(result.device, "cuda")

    def test_unreadable_gpu_info_does_not_block_inference(self) -> None:
        with self._patched_runtime(cuda_available=True) as fake:

            def broken_arch_list() -> list[str]:
                raise RuntimeError("no driver")

            fake.torch.cuda.get_arch_list = broken_arch_list
            result = birefnet_inference.predict_alpha(
                get_model_spec("birefnet_hr_matting"),
                Path("/tmp/model"),
                self._runtime_ready(),
                object(),
            )

        self.assertEqual(result.device, "cuda")

    def test_cuda_device_support_follows_torch_major_version_rule(self) -> None:
        def supported(capability: tuple[int, int], arch_list: tuple[str, ...]) -> bool:
            device = birefnet_inference.cuda_device(
                _FakeTorch(True, capability=capability, arch_list=arch_list)
            )
            assert device is not None
            return device.supported

        self.assertTrue(supported((8, 9), ("sm_80", "sm_86", "sm_90")))
        self.assertTrue(supported((10, 0), ("sm_90", "sm_100", "sm_120")))
        self.assertTrue(supported((12, 0), ("sm_120",)))
        self.assertTrue(supported((12, 0), ("compute_120",)))
        self.assertFalse(supported((12, 0), ("sm_50", "sm_90")))
        self.assertFalse(supported((10, 0), ("sm_80", "sm_90")))

    def test_supported_gpu_is_logged_with_arch_list(self) -> None:
        stderr = io.StringIO()
        with (
            self._patched_runtime(
                cuda_available=True, capability=(12, 0), arch_list=("sm_90", "sm_120")
            ),
            contextlib.redirect_stderr(stderr),
        ):
            birefnet_inference.predict_alpha(
                get_model_spec("birefnet_hr_matting"),
                Path("/tmp/model"),
                self._runtime_ready(),
                object(),
            )

        log = stderr.getvalue()
        self.assertIn("event.name=alpha_model.cuda.checked", log)
        self.assertIn("device.compute_capability=12.0", log)
        self.assertIn('torch.cuda.arch_list=["sm_90", "sm_120"]', log)

    def test_no_kernel_image_error_maps_to_unsupported_gpu(self) -> None:
        error = RuntimeError("CUDA error: no kernel image is available for execution on the device")
        with (
            self._patched_runtime(cuda_available=True, model_error=error),
            self.assertRaises(birefnet_inference.InferenceUnsupportedGpu) as raised,
        ):
            birefnet_inference.predict_alpha(
                get_model_spec("birefnet_hr_matting"),
                Path("/tmp/model"),
                self._runtime_ready(),
                object(),
            )

        self.assertIn("no CUDA kernels", str(raised.exception))

    def test_inference_failure_names_device_and_cause_and_logs_reason(self) -> None:
        stderr = io.StringIO()
        with (
            self._patched_runtime(
                cuda_available=True, model_error=RuntimeError("shape mismatch in decoder")
            ),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(birefnet_inference.InferenceFailed) as raised,
        ):
            birefnet_inference.predict_alpha(
                get_model_spec("birefnet_hr_matting"),
                Path("/tmp/model"),
                self._runtime_ready(),
                object(),
            )

        self.assertEqual(str(raised.exception), "alpha extraction failed (cuda, RuntimeError)")
        log = stderr.getvalue()
        self.assertIn("event.name=alpha_model.predict.failed", log)
        self.assertIn("error.type=RuntimeError", log)
        self.assertIn("shape mismatch in decoder", log)

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

    def test_release_runtime_memory_clears_model_cache_and_cuda_allocator(self) -> None:
        with self._patched_runtime(cuda_available=True) as fake:
            spec = get_model_spec("birefnet_hr_matting")
            birefnet_inference.predict_alpha(
                spec, Path("/tmp/model"), self._runtime_ready(), object()
            )

            self.assertEqual(len(birefnet_inference._MODEL_CACHE), 1)
            birefnet_inference.release_runtime_memory()

        self.assertEqual(len(birefnet_inference._MODEL_CACHE), 0)
        self.assertEqual(fake.models[0].device, "cpu")
        self.assertEqual(fake.torch.empty_cache_calls, 1)
        self.assertEqual(fake.torch.ipc_collect_calls, 1)

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

    def _patched_runtime(
        self,
        cuda_available: bool,
        raise_oom: bool = False,
        capability: tuple[int, int] = (8, 9),
        arch_list: tuple[str, ...] = ("sm_80", "sm_86", "sm_90"),
        model_error: Exception | None = None,
    ) -> _RuntimePatch:
        return _RuntimePatch(cuda_available, raise_oom, capability, arch_list, model_error)


class _RuntimePatch:
    def __init__(
        self,
        cuda_available: bool,
        raise_oom: bool,
        capability: tuple[int, int],
        arch_list: tuple[str, ...],
        model_error: Exception | None,
    ) -> None:
        self.cuda_available = cuda_available
        self.raise_oom = raise_oom
        self.model_error = model_error
        self.from_pretrained_calls: list[dict[str, Any]] = []
        self.resize_sizes: list[tuple[int, int]] = []
        self.models: list[_FakeModel] = []
        self.torch = _FakeTorch(cuda_available, capability=capability, arch_list=arch_list)
        self._original_import_module = birefnet_inference.importlib.import_module

    def __enter__(self) -> _RuntimePatch:
        birefnet_inference.importlib.import_module = self._fake_import_module
        return self

    def __exit__(self, *_args: object) -> None:
        birefnet_inference.importlib.import_module = self._original_import_module

    def _fake_import_module(self, name: str) -> object:
        if name == "torch":
            return self.torch
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
                    model = _FakeModel(patch.raise_oom, patch.model_error)
                    patch.models.append(model)
                    return model

            return types.SimpleNamespace(
                AutoModelForImageSegmentation=AutoModelForImageSegmentation
            )
        return self._original_import_module(name)


class _FakeTorch:
    __version__ = "2.6.0"

    def __init__(
        self,
        cuda_available: bool,
        capability: tuple[int, int] = (8, 9),
        arch_list: tuple[str, ...] = ("sm_80", "sm_86", "sm_90"),
    ) -> None:
        self.empty_cache_calls = 0
        self.ipc_collect_calls = 0
        self.cuda = types.SimpleNamespace(
            is_available=lambda: cuda_available,
            get_device_capability=lambda _index=0: capability,
            get_arch_list=lambda: list(arch_list),
            get_device_name=lambda _index=0: "Fake GPU",
            memory_allocated=lambda: 123,
            memory_reserved=lambda: 456,
            empty_cache=self._empty_cache,
            ipc_collect=self._ipc_collect,
        )

    def _empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def _ipc_collect(self) -> None:
        self.ipc_collect_calls += 1

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
    def __init__(self, raise_oom: bool, error: Exception | None = None) -> None:
        self.raise_oom = raise_oom
        self.error = error
        self.device = ""

    def to(self, device: str) -> _FakeModel:
        self.device = device
        return self

    def eval(self) -> _FakeModel:
        return self

    def half(self) -> _FakeModel:
        return self

    def __call__(self, _tensor: object) -> list[_FakePrediction]:
        if self.raise_oom:
            raise RuntimeError("CUDA out of memory")
        if self.error is not None:
            raise self.error
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
