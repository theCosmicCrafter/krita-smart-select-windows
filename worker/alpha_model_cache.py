from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from alpha_model_registry import ModelSpec
from job_runner import JobContext
from observability import log_event

CacheState = Literal["missing", "partial", "verified", "invalid"]
ProgressCallback = Callable[[str, float | None, dict[str, object]], None]


class ModelCacheError(RuntimeError):
    pass


class ModelCacheDirMissing(ModelCacheError):
    pass


class ModelDownloadCancelled(ModelCacheError):
    pass


class ModelDownloadError(ModelCacheError):
    pass


class ModelVerificationError(ModelCacheError):
    pass


@dataclass(frozen=True)
class ModelCacheStatus:
    model_id: str
    snapshot_dir: Path | None
    state: CacheState
    checkpoint_bytes: int | None
    config_bytes: int | None


class ModelCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def status(self, spec: ModelSpec) -> ModelCacheStatus:
        model_dir = self._model_dir(spec)
        snapshot_dir = model_dir / "snapshot"
        checkpoint_path = snapshot_dir / spec.checkpoint_filename
        config_path = snapshot_dir / spec.config_filename
        checkpoint_bytes = _file_size(checkpoint_path)
        config_bytes = _file_size(config_path)
        if not snapshot_dir.exists():
            return ModelCacheStatus(spec.id, None, "missing", None, None)
        if (
            checkpoint_bytes == spec.checkpoint_expected_bytes
            and config_bytes == spec.config_expected_bytes
        ):
            return ModelCacheStatus(
                spec.id, snapshot_dir, "verified", checkpoint_bytes, config_bytes
            )
        state: CacheState = (
            "partial" if checkpoint_bytes is None or config_bytes is None else "invalid"
        )
        return ModelCacheStatus(spec.id, snapshot_dir, state, checkpoint_bytes, config_bytes)

    def ensure_downloaded(
        self,
        spec: ModelSpec,
        progress: ProgressCallback,
        context: JobContext | None = None,
    ) -> ModelCacheStatus:
        if _is_plugin_source_path(self.root):
            raise ModelCacheDirMissing("model cache is not writable")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ModelCacheDirMissing("model cache is not writable") from exc
        if not self.root.is_dir():
            raise ModelCacheDirMissing("model cache is not writable")

        status = self.status(spec)
        progress("checking_model_cache", None, {"model.cache.state": status.state})
        if status.state == "verified":
            log_event(
                "alpha_model.cache.result",
                component="model_cache",
                **{
                    "model.id": spec.id,
                    "model.cache.state": "verified",
                    "model.checkpoint.bytes": status.checkpoint_bytes,
                },
            )
            return status

        _check_cancelled(context)
        if status.state == "invalid":
            shutil.rmtree(self._model_dir(spec), ignore_errors=True)

        progress("downloading_model", None, {})
        log_event(
            "alpha_model.cache.start",
            component="model_cache",
            **{"model.id": spec.id, "model.repo_id": spec.repo_id, "model.revision": spec.revision},
        )
        try:
            snapshot_dir = self._snapshot_download(spec)
        except Exception as exc:
            log_event(
                "alpha_model.cache.failed",
                severity="error",
                component="model_cache",
                **{
                    "model.id": spec.id,
                    "error.type": exc.__class__.__name__,
                    "error.message": str(exc),
                },
            )
            raise ModelDownloadError("model download failed") from exc
        _check_cancelled(context)

        progress("verifying_model", None, {})
        verified = self._verify(spec, snapshot_dir)
        self._write_manifest(spec, verified)
        log_event(
            "alpha_model.cache.result",
            component="model_cache",
            **{
                "model.id": spec.id,
                "model.cache.state": "verified",
                "model.checkpoint.bytes": verified.checkpoint_bytes,
            },
        )
        return verified

    def _snapshot_download(self, spec: ModelSpec) -> Path:
        huggingface_hub = __import__("huggingface_hub")
        model_dir = self._model_dir(spec)
        snapshot_dir = model_dir / "snapshot"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        huggingface_hub.snapshot_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            local_dir=snapshot_dir,
            local_dir_use_symlinks=False,
            allow_patterns=["*.py", "*.json", "*.safetensors", "README.md"],
        )
        return snapshot_dir

    def _verify(self, spec: ModelSpec, snapshot_dir: Path) -> ModelCacheStatus:
        checkpoint_bytes = _file_size(snapshot_dir / spec.checkpoint_filename)
        config_bytes = _file_size(snapshot_dir / spec.config_filename)
        if checkpoint_bytes != spec.checkpoint_expected_bytes:
            raise ModelVerificationError("model verification failed")
        if config_bytes != spec.config_expected_bytes:
            raise ModelVerificationError("model verification failed")
        return ModelCacheStatus(spec.id, snapshot_dir, "verified", checkpoint_bytes, config_bytes)

    def _write_manifest(self, spec: ModelSpec, status: ModelCacheStatus) -> None:
        payload = {
            "model_id": spec.id,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "license": spec.license,
            "checkpoint_bytes": status.checkpoint_bytes,
            "config_bytes": status.config_bytes,
        }
        (self._model_dir(spec) / "model.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _model_dir(self, spec: ModelSpec) -> Path:
        return self.root / spec.id


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _check_cancelled(context: JobContext | None) -> None:
    if context is not None and context.is_cancelled():
        raise ModelDownloadCancelled("model download cancelled")


def _is_plugin_source_path(root: Path) -> bool:
    try:
        worker_dir = Path(__file__).resolve().parent
        return root.resolve().is_relative_to(worker_dir.parent.resolve())
    except OSError:
        return True
