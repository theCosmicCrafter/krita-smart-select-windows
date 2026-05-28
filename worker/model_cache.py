from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from http.client import HTTPMessage
from pathlib import Path
from typing import Literal, Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from model_registry import ModelSpec
from observability import log_event

CacheState = Literal["missing", "partial", "verified", "invalid"]

CHUNK_SIZE = 12 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 30.0
DOWNLOAD_RETRY_ATTEMPTS = 3


class ModelCacheError(RuntimeError):
    pass


class ModelCacheDirMissing(ModelCacheError):
    pass


class ModelDownloadError(ModelCacheError):
    pass


class ModelVerificationError(ModelCacheError):
    pass


class ModelDownloadCancelled(ModelCacheError):
    pass


@dataclass(frozen=True)
class ModelCacheStatus:
    model_id: str
    checkpoint_path: Path | None
    partial_path: Path | None
    state: CacheState
    expected_bytes: int | None
    actual_bytes: int | None


class ResponseLike(Protocol):
    headers: HTTPMessage

    def read(self, size: int = -1) -> bytes: ...


class ResponseContextLike(Protocol):
    def __enter__(self) -> ResponseLike: ...
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> object: ...


class UrlOpenLike(Protocol):
    def __call__(self, request: Request, *, timeout: float) -> ResponseContextLike: ...


ProgressCallback = Callable[[str, float | None, dict[str, object]], None]
CancelCallback = Callable[[], bool]


def _default_urlopen(request: Request, *, timeout: float) -> ResponseContextLike:
    return cast(ResponseContextLike, urlopen(request, timeout=timeout))


def _exception_reason(exc: BaseException) -> str:
    if exc.__cause__ is None:
        return repr(exc)
    return f"{exc!r}; cause={exc.__cause__!r}"


class ModelCache:
    def __init__(self, root: Path, opener: UrlOpenLike | None = None) -> None:
        self._root = root
        self._opener = opener or _default_urlopen

    def status(self, spec: ModelSpec) -> ModelCacheStatus:
        if not spec.requires_download:
            return ModelCacheStatus(
                model_id=spec.id,
                checkpoint_path=None,
                partial_path=None,
                state="verified",
                expected_bytes=spec.expected_bytes,
                actual_bytes=None,
            )

        checkpoint_path = self.checkpoint_path(spec)
        partial_path = self.partial_path(spec)
        expected_bytes = spec.expected_bytes
        if checkpoint_path.exists():
            actual_bytes = checkpoint_path.stat().st_size
            if expected_bytes is None or actual_bytes == expected_bytes:
                return ModelCacheStatus(
                    model_id=spec.id,
                    checkpoint_path=checkpoint_path,
                    partial_path=partial_path,
                    state="verified",
                    expected_bytes=expected_bytes,
                    actual_bytes=actual_bytes,
                )
            return ModelCacheStatus(
                model_id=spec.id,
                checkpoint_path=checkpoint_path,
                partial_path=partial_path,
                state="invalid",
                expected_bytes=expected_bytes,
                actual_bytes=actual_bytes,
            )

        if partial_path.exists():
            return ModelCacheStatus(
                model_id=spec.id,
                checkpoint_path=checkpoint_path,
                partial_path=partial_path,
                state="partial",
                expected_bytes=expected_bytes,
                actual_bytes=partial_path.stat().st_size,
            )

        return ModelCacheStatus(
            model_id=spec.id,
            checkpoint_path=checkpoint_path,
            partial_path=partial_path,
            state="missing",
            expected_bytes=expected_bytes,
            actual_bytes=None,
        )

    def ensure_downloaded(
        self,
        spec: ModelSpec,
        progress: ProgressCallback,
        is_cancelled: CancelCallback,
    ) -> ModelCacheStatus:
        if not spec.requires_download:
            return self.status(spec)

        self._ensure_model_dir(spec)
        status = self.status(spec)
        if status.state == "verified":
            self._write_sidecar(spec, status)
            return status

        if status.state == "invalid" and status.checkpoint_path is not None:
            self._move_invalid_checkpoint(status.checkpoint_path)

        if is_cancelled():
            raise ModelDownloadCancelled("model download cancelled")

        self._download(spec, progress, is_cancelled)
        verified = self.status(spec)
        if verified.state != "verified":
            raise ModelVerificationError("model verification failed")
        self._write_sidecar(spec, verified)
        return verified

    def checkpoint_path(self, spec: ModelSpec) -> Path:
        if spec.checkpoint_filename is None:
            raise ModelVerificationError("model checkpoint filename is missing")
        return self._model_dir(spec) / spec.checkpoint_filename

    def partial_path(self, spec: ModelSpec) -> Path:
        return self.checkpoint_path(spec).with_name(self.checkpoint_path(spec).name + ".partial")

    def _model_dir(self, spec: ModelSpec) -> Path:
        return self._root / spec.id

    def _ensure_model_dir(self, spec: ModelSpec) -> None:
        try:
            self._model_dir(spec).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ModelCacheDirMissing("model cache is not writable") from exc
        if not os.access(self._model_dir(spec), os.W_OK):
            raise ModelCacheDirMissing("model cache is not writable")

    def _move_invalid_checkpoint(self, checkpoint_path: Path) -> None:
        invalid_path = checkpoint_path.with_name(checkpoint_path.name + ".invalid")
        invalid_path.unlink(missing_ok=True)
        checkpoint_path.replace(invalid_path)

    def _download(
        self,
        spec: ModelSpec,
        progress: ProgressCallback,
        is_cancelled: CancelCallback,
    ) -> None:
        if spec.checkpoint_url is None or spec.expected_bytes is None:
            raise ModelDownloadError("model download failed")

        checkpoint_path = self.checkpoint_path(spec)
        partial_path = self.partial_path(spec)
        start_offset = partial_path.stat().st_size if partial_path.exists() else 0
        if start_offset > spec.expected_bytes:
            partial_path.unlink()
            start_offset = 0
        self._ensure_disk_space(partial_path.parent, spec.id, spec.expected_bytes - start_offset)

        log_event(
            "model.download.start",
            component="model_cache",
            **{"model.id": spec.id, "download.bytes_done": start_offset},
        )
        try:
            try:
                downloaded = self._download_once(
                    spec, partial_path, start_offset, progress, is_cancelled
                )
            except ModelDownloadError:
                if start_offset <= 0:
                    raise
                partial_path.unlink(missing_ok=True)
                downloaded = self._download_once(spec, partial_path, 0, progress, is_cancelled)
        except ModelDownloadCancelled:
            raise
        except ModelDownloadError as exc:
            log_event(
                "model.download.failed",
                severity="error",
                component="model_cache",
                **{
                    "model.id": spec.id,
                    "error.type": exc.__class__.__name__,
                    "error.reason": _exception_reason(exc),
                },
            )
            raise
        except (HTTPError, TimeoutError, URLError, OSError) as exc:
            log_event(
                "model.download.failed",
                severity="error",
                component="model_cache",
                **{
                    "model.id": spec.id,
                    "error.type": exc.__class__.__name__,
                    "error.reason": _exception_reason(exc),
                },
            )
            raise ModelDownloadError("model download failed") from exc

        if downloaded != spec.expected_bytes:
            log_event(
                "model.download.failed",
                severity="error",
                component="model_cache",
                **{
                    "model.id": spec.id,
                    "error.type": "ModelVerificationError",
                    "download.bytes_done": downloaded,
                },
            )
            raise ModelVerificationError("model verification failed")

        partial_path.replace(checkpoint_path)
        log_event(
            "model.download.result",
            component="model_cache",
            **{
                "model.id": spec.id,
                "model.cache.state": "verified",
                "model.checkpoint.bytes": downloaded,
            },
        )

    def _download_once(
        self,
        spec: ModelSpec,
        partial_path: Path,
        start_offset: int,
        progress: ProgressCallback,
        is_cancelled: CancelCallback,
    ) -> int:
        if spec.checkpoint_url is None or spec.expected_bytes is None:
            raise ModelDownloadError("model download failed")

        mode = "ab" if start_offset > 0 else "wb"
        downloaded = start_offset
        with partial_path.open(mode) as file:
            while downloaded < spec.expected_bytes:
                if is_cancelled():
                    raise ModelDownloadCancelled("model download cancelled")

                range_end = min(downloaded + CHUNK_SIZE - 1, spec.expected_bytes - 1)
                chunk = self._read_range_with_retries(spec, downloaded, range_end, is_cancelled)
                file.write(chunk)
                downloaded += len(chunk)
                progress_value = min(downloaded / spec.expected_bytes, 1.0)
                progress(
                    "downloading_model",
                    progress_value,
                    {
                        "bytes_done": downloaded,
                        "bytes_total": spec.expected_bytes,
                    },
                )
                log_event(
                    "model.download.progress",
                    component="model_cache",
                    **{
                        "model.id": spec.id,
                        "download.bytes_done": downloaded,
                        "download.bytes_total": spec.expected_bytes,
                    },
                )
        return downloaded

    def _read_range_with_retries(
        self,
        spec: ModelSpec,
        range_start: int,
        range_end: int,
        is_cancelled: CancelCallback,
    ) -> bytes:
        if spec.checkpoint_url is None:
            raise ModelDownloadError("model download failed")

        expected_length = range_end - range_start + 1
        last_error: Exception | None = None
        for attempt in range(1, DOWNLOAD_RETRY_ATTEMPTS + 1):
            if is_cancelled():
                raise ModelDownloadCancelled("model download cancelled")
            request = Request(spec.checkpoint_url)
            request.add_header("Range", f"bytes={range_start}-{range_end}")
            try:
                with self._opener(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                    if response.headers.get("Content-Range") is None:
                        raise ModelDownloadError("model download failed: missing content-range")
                    chunk = response.read(expected_length)
                    if len(chunk) != expected_length:
                        raise ModelDownloadError("model download failed: short read")
                    return chunk
            except ModelDownloadError as exc:
                last_error = exc
                break
            except (HTTPError, TimeoutError, URLError, OSError) as exc:
                last_error = exc
                if attempt == DOWNLOAD_RETRY_ATTEMPTS:
                    break
                log_event(
                    "model.download.retry",
                    severity="warning",
                    component="model_cache",
                    **{
                        "model.id": spec.id,
                        "retry.attempt": attempt,
                        "error.type": exc.__class__.__name__,
                        "error.reason": _exception_reason(exc),
                        "download.range_start": range_start,
                        "download.range_end": range_end,
                    },
                )

        raise ModelDownloadError("model download failed") from last_error

    def _ensure_disk_space(self, directory: Path, model_id: str, required_bytes: int) -> None:
        free_bytes = shutil.disk_usage(directory).free
        if free_bytes < required_bytes:
            log_event(
                "model.download.failed",
                severity="error",
                component="model_cache",
                **{
                    "model.id": model_id,
                    "error.type": "InsufficientDiskSpace",
                    "error.reason": "insufficient disk space",
                    "disk.bytes_required": required_bytes,
                    "disk.bytes_free": free_bytes,
                },
            )
            raise ModelDownloadError("model download failed")

    def _write_sidecar(self, spec: ModelSpec, status: ModelCacheStatus) -> None:
        sidecar_path = self._model_dir(spec) / "model.json"
        payload = {
            "id": spec.id,
            "title": spec.title,
            "checkpoint_filename": spec.checkpoint_filename,
            "expected_bytes": spec.expected_bytes,
            "actual_bytes": status.actual_bytes,
            "etag": spec.etag,
            "cache_state": status.state,
        }
        sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
