from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

WORKER_DIR = Path(__file__).resolve().parents[2] / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from worker.alpha_model_cache import ModelCacheStatus  # noqa: E402
from worker.alpha_model_registry import get_model_spec  # noqa: E402
from worker.job_runner import JobContext  # noqa: E402
from worker.region_selection import RegionSelectionInput  # noqa: E402

from worker import birefnet_inference, runtime_env, smart_select_worker  # noqa: E402


class SmartSelectWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.original_emit = smart_select_worker.emit
        smart_select_worker.emit = self.messages.append

    def tearDown(self) -> None:
        smart_select_worker.emit = self.original_emit

    def test_prepare_selection_succeeds_with_fake_birefnet_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            self._with_fake_pipeline(temp, lambda: self._run_prepare_selection(temp))

            result = self.messages[-1]["result"]
            mask_path = Path(result["selection_mask_path"])
            self.assertTrue(mask_path.exists())
            self.assertEqual(result["model"]["id"], "birefnet_hr_matting")
            self.assertEqual(result["runtime"]["device"], "cuda")

            stages = [
                message.get("stage")
                for message in self.messages
                if message.get("event") == "progress"
            ]
            self.assertIn("checking_runtime", stages)
            self.assertIn("running_alpha_model", stages)
            self.assertIn("building_alpha_mask", stages)

    def test_missing_model_cache_dir_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            params = self._params(temp)
            params.pop("model_cache_dir")
            self._with_fake_runtime(
                lambda: smart_select_worker.handle_prepare_selection(7, params, self._context())
            )

        self.assertEqual(self.messages[-1]["error"]["type"], "ModelCacheDirMissing")

    def test_missing_runtime_cache_dir_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            params = self._params(temp)
            params.pop("runtime_cache_dir")
            self._with_region_loader_that_must_not_run(
                lambda: smart_select_worker.handle_prepare_selection(7, params, self._context())
            )

        self.assertEqual(self.messages[-1]["error"]["type"], "RuntimeCacheDirMissing")

    def test_missing_runtime_cache_dir_does_not_load_region(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            params = self._params(temp)
            params.pop("runtime_cache_dir")

            self._with_region_loader_that_must_not_run(
                lambda: smart_select_worker.handle_prepare_selection(7, params, self._context())
            )

        self.assertEqual(self.messages[-1]["error"]["type"], "RuntimeCacheDirMissing")

    def test_unknown_model_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            params = self._params(temp)
            params["model_id"] = "mock"
            self._with_runtime_that_must_not_run(
                lambda: smart_select_worker.handle_prepare_selection(7, params, self._context())
            )

        self.assertEqual(self.messages[-1]["error"]["type"], "UnknownModel")

    def test_unknown_model_does_not_setup_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            params = self._params(temp)
            params["model_id"] = "not_a_real_model"

            self._with_runtime_that_must_not_run(
                lambda: smart_select_worker.handle_prepare_selection(7, params, self._context())
            )

        self.assertEqual(self.messages[-1]["error"]["type"], "UnknownModel")

    def test_runtime_is_ensured_before_region_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            runtime_ready = False
            original_runtime = smart_select_worker.runtime_env.ensure_runtime
            original_region = smart_select_worker.region_selection.load_region_selection_input
            original_cache = smart_select_worker.ModelCache
            original_predict = smart_select_worker.birefnet_inference.predict_alpha

            def fake_runtime(*args: Any, **kwargs: Any) -> runtime_env.RuntimeReady:
                nonlocal runtime_ready
                runtime_ready = True
                return self._fake_runtime(*args, **kwargs)

            def fake_region(*_args: Any, **_kwargs: Any) -> RegionSelectionInput:
                self.assertTrue(runtime_ready)
                return self._region()

            try:
                smart_select_worker.runtime_env.ensure_runtime = fake_runtime
                smart_select_worker.region_selection.load_region_selection_input = fake_region
                smart_select_worker.ModelCache = lambda _root: _FakeModelCache(temp)
                smart_select_worker.birefnet_inference.predict_alpha = self._fake_predict
                self._run_prepare_selection(temp)
            finally:
                smart_select_worker.runtime_env.ensure_runtime = original_runtime
                smart_select_worker.region_selection.load_region_selection_input = original_region
                smart_select_worker.ModelCache = original_cache
                smart_select_worker.birefnet_inference.predict_alpha = original_predict

        self.assertIn("result", self.messages[-1])

    def test_inference_oom_returns_user_safe_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)

            def run() -> None:
                original_predict = smart_select_worker.birefnet_inference.predict_alpha
                try:
                    smart_select_worker.birefnet_inference.predict_alpha = self._raise_oom
                    self._run_prepare_selection(temp)
                finally:
                    smart_select_worker.birefnet_inference.predict_alpha = original_predict

            self._with_fake_pipeline(temp, run, patch_predict=False)

        self.assertEqual(self.messages[-1]["error"]["type"], "InferenceOutOfMemory")

    def test_unsupported_gpu_error_names_the_cause(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)

            def run() -> None:
                original_predict = smart_select_worker.birefnet_inference.predict_alpha
                try:
                    smart_select_worker.birefnet_inference.predict_alpha = (
                        self._raise_unsupported_gpu
                    )
                    self._run_prepare_selection(temp)
                finally:
                    smart_select_worker.birefnet_inference.predict_alpha = original_predict

            self._with_fake_pipeline(temp, run, patch_predict=False)

        error = self.messages[-1]["error"]
        self.assertEqual(error["type"], "InferenceUnsupportedGpu")
        self.assertIn("Fake GPU", error["message"])

    def test_cancel_before_inference_emits_no_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            context = self._context(cancelled=True)
            self._with_fake_pipeline(
                temp,
                lambda: smart_select_worker.handle_prepare_selection(
                    7, self._params(temp), context
                ),
            )

        self.assertFalse(any("result" in message for message in self.messages))

    def test_dispatch_releases_birefnet_memory_after_selection_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            release_calls = 0
            original_release = smart_select_worker.birefnet_inference.release_runtime_memory

            def fake_release() -> None:
                nonlocal release_calls
                release_calls += 1

            try:
                smart_select_worker.birefnet_inference.release_runtime_memory = fake_release
                self._with_fake_pipeline(
                    temp,
                    lambda: smart_select_worker.dispatch_job(
                        self._context(),
                        {"_method": "prepare_selection", **self._params(temp)},
                    ),
                )
            finally:
                smart_select_worker.birefnet_inference.release_runtime_memory = original_release

        self.assertEqual(release_calls, 1)
        self.assertIn("result", self.messages[-1])

    def test_debug_export_writes_only_birefnet_debug_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            original_environ = dict(smart_select_worker.os.environ)
            try:
                smart_select_worker.os.environ["SMART_SELECT_KEEP_DEBUG_FILES"] = "1"
                self._with_fake_pipeline(temp, lambda: self._run_prepare_selection(temp))
            finally:
                smart_select_worker.os.environ.clear()
                smart_select_worker.os.environ.update(original_environ)

        debug_events = [
            message
            for message in self.messages
            if message.get("event.name") == "selection.debug_export.result"
        ]
        self.assertEqual(debug_events, [])
        # The worker logs debug export paths to stderr, not protocol stdout. This test mainly
        # proves the debug path does not add legacy SAM/MAM fields to the result payload.
        self.assertNotIn("refiner", self.messages[-1]["result"])
        self.assertNotIn("segments_path", self.messages[-1]["result"])

    def _run_prepare_selection(self, temp: Path) -> None:
        smart_select_worker.handle_prepare_selection(7, self._params(temp), self._context())

    def _params(self, temp: Path) -> smart_select_worker.PrepareSelectionParams:
        return {
            "image_path": str(temp / "snapshot.png"),
            "alpha_mask_path": str(temp / "alpha.npy"),
            "region_mask_path": str(temp / "region.npy"),
            "region_bbox": [0, 0, 2, 2],
            "model_id": "birefnet_hr_matting",
            "algorithm_version": "birefnet-hr-matting-lasso-v1",
            "fingerprint": "fixture",
            "canvas_width": 2,
            "canvas_height": 2,
            "model_cache_dir": str(temp / "models"),
            "runtime_cache_dir": str(temp / "runtime"),
        }

    @staticmethod
    def _context(cancelled: bool = False) -> JobContext:
        context = JobContext(job_id=7)
        if cancelled:
            context.cancel()
        return context

    def _with_fake_region(self, callback: Any) -> None:
        original_region = smart_select_worker.region_selection.load_region_selection_input
        try:
            smart_select_worker.region_selection.load_region_selection_input = (
                lambda *_args, **_kwargs: self._region()
            )
            callback()
        finally:
            smart_select_worker.region_selection.load_region_selection_input = original_region

    def _with_region_loader_that_must_not_run(self, callback: Any) -> None:
        original_region = smart_select_worker.region_selection.load_region_selection_input
        try:
            smart_select_worker.region_selection.load_region_selection_input = (
                self._fail_region_load
            )
            callback()
        finally:
            smart_select_worker.region_selection.load_region_selection_input = original_region

    def _with_runtime_that_must_not_run(self, callback: Any) -> None:
        original_runtime = smart_select_worker.runtime_env.ensure_runtime
        try:
            smart_select_worker.runtime_env.ensure_runtime = self._fail_runtime_setup
            callback()
        finally:
            smart_select_worker.runtime_env.ensure_runtime = original_runtime

    def _with_fake_runtime(self, callback: Any) -> None:
        original_runtime = smart_select_worker.runtime_env.ensure_runtime
        try:
            smart_select_worker.runtime_env.ensure_runtime = self._fake_runtime
            self._with_fake_region(callback)
        finally:
            smart_select_worker.runtime_env.ensure_runtime = original_runtime

    def _with_fake_pipeline(self, temp: Path, callback: Any, *, patch_predict: bool = True) -> None:
        original_cache = smart_select_worker.ModelCache
        original_runtime = smart_select_worker.runtime_env.ensure_runtime
        original_predict = smart_select_worker.birefnet_inference.predict_alpha
        try:
            smart_select_worker.ModelCache = lambda _root: _FakeModelCache(temp)
            smart_select_worker.runtime_env.ensure_runtime = self._fake_runtime
            if patch_predict:
                smart_select_worker.birefnet_inference.predict_alpha = self._fake_predict
            self._with_fake_region(callback)
        finally:
            smart_select_worker.ModelCache = original_cache
            smart_select_worker.runtime_env.ensure_runtime = original_runtime
            smart_select_worker.birefnet_inference.predict_alpha = original_predict

    @staticmethod
    def _region() -> RegionSelectionInput:
        return RegionSelectionInput(
            crop_image_array=object(),
            crop_alpha_mask=b"\xff" * 4,
            crop_region_mask=b"\xff" * 4,
            crop_bbox=(0, 0, 2, 2),
            region_bbox=(0, 0, 2, 2),
            prompt_box_xyxy=(0.0, 0.0, 2.0, 2.0),
            point_coords=(),
            point_labels=(),
            canvas_width=2,
            canvas_height=2,
        )

    @staticmethod
    def _fake_runtime(*args: Any, **_kwargs: Any) -> runtime_env.RuntimeReady:
        if len(args) >= 2 and callable(args[1]):
            args[1]("checking_runtime", None, {})
        return runtime_env.RuntimeReady(
            root=Path("/tmp/runtime"),
            venv_path=Path("/tmp/runtime/venv"),
            python_executable=Path("/tmp/runtime/venv/bin/python"),
            site_packages=Path("/tmp/runtime/venv/site-packages"),
            torch_version="2.6.0",
            cuda_available=True,
        )

    @staticmethod
    def _fake_predict(*_args: Any, **_kwargs: Any) -> birefnet_inference.BiRefNetResult:
        return birefnet_inference.BiRefNetResult(
            crop_alpha=bytes([0, 128, 255, 64]),
            width=2,
            height=2,
            device="cuda",
            torch_version="2.6.0",
            cuda_available=True,
            duration_ms=12,
        )

    @staticmethod
    def _raise_oom(*_args: Any, **_kwargs: Any) -> birefnet_inference.BiRefNetResult:
        raise smart_select_worker.birefnet_inference.InferenceOutOfMemory(
            "alpha extraction ran out of GPU memory"
        )

    @staticmethod
    def _raise_unsupported_gpu(*_args: Any, **_kwargs: Any) -> birefnet_inference.BiRefNetResult:
        raise smart_select_worker.birefnet_inference.InferenceUnsupportedGpu(
            "Fake GPU (compute capability 12.0) is not supported by the installed PyTorch"
        )

    @staticmethod
    def _fail_region_load(*_args: Any, **_kwargs: Any) -> RegionSelectionInput:
        raise AssertionError("region loading should not run")

    @staticmethod
    def _fail_runtime_setup(*_args: Any, **_kwargs: Any) -> runtime_env.RuntimeReady:
        raise AssertionError("runtime setup should not run")


class _FakeModelCache:
    def __init__(self, temp: Path) -> None:
        self.temp = temp

    def ensure_downloaded(self, *_args: Any, **_kwargs: Any) -> ModelCacheStatus:
        spec = get_model_spec("birefnet_hr_matting")
        snapshot = self.temp / "snapshot"
        snapshot.mkdir(exist_ok=True)
        return ModelCacheStatus(
            model_id=spec.id,
            snapshot_dir=snapshot,
            state="verified",
            checkpoint_bytes=spec.checkpoint_expected_bytes,
            config_bytes=spec.config_expected_bytes,
        )


def parse_worker_output(line: str) -> dict[str, Any]:
    return json.loads(line)


if __name__ == "__main__":
    unittest.main()
