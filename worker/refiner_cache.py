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

from observability import log_event
from refiner_registry import RefinerSpec

CacheState = Literal["missing", "partial", "verified", "invalid"]

CHUNK_SIZE = 12 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 30.0
DOWNLOAD_RETRY_ATTEMPTS = 3


class RefinerCacheError(RuntimeError):
    pass


class RefinerCacheDirMissing(RefinerCacheError):
    pass


class RefinerDownloadError(RefinerCacheError):
    pass


class RefinerVerificationError(RefinerCacheError):
    pass


class RefinerDownloadCancelled(RefinerCacheError):
    pass


class RefinerUnavailable(RefinerCacheError):
    pass


@dataclass(frozen=True)
class RefinerCacheStatus:
    refiner_id: str
    state: CacheState
    sam_checkpoint_path: Path | None
    mam_checkpoint_path: Path | None
    sam_actual_bytes: int | None
    mam_actual_bytes: int | None


@dataclass(frozen=True)
class _DownloadSpec:
    role: Literal["sam", "mam"]
    url: str
    filename: str
    expected_bytes: int
    etag: str | None


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


class RefinerCache:
    def __init__(self, root: Path, opener: UrlOpenLike | None = None) -> None:
        self._root = root
        self._opener = opener or _default_urlopen

    def status(self, spec: RefinerSpec) -> RefinerCacheStatus:
        if not spec.requires_download:
            return RefinerCacheStatus(
                refiner_id=spec.id,
                state="verified",
                sam_checkpoint_path=None,
                mam_checkpoint_path=None,
                sam_actual_bytes=None,
                mam_actual_bytes=None,
            )

        sam_status = self._file_status(spec, self._sam_download_spec(spec))
        mam_status = self._file_status(spec, self._mam_download_spec(spec))
        if sam_status[0] == "verified" and mam_status[0] == "verified":
            state: CacheState = "verified"
        elif sam_status[0] == "invalid" or mam_status[0] == "invalid":
            state = "invalid"
        elif sam_status[0] == "partial" or mam_status[0] == "partial":
            state = "partial"
        else:
            state = "missing"

        return RefinerCacheStatus(
            refiner_id=spec.id,
            state=state,
            sam_checkpoint_path=self.checkpoint_path(spec, "sam"),
            mam_checkpoint_path=self.checkpoint_path(spec, "mam"),
            sam_actual_bytes=sam_status[1],
            mam_actual_bytes=mam_status[1],
        )

    def ensure_downloaded(
        self,
        spec: RefinerSpec,
        progress: ProgressCallback,
        is_cancelled: CancelCallback,
    ) -> RefinerCacheStatus:
        if not spec.requires_download:
            return self.status(spec)

        self._validate_spec(spec)
        self._ensure_refiner_dir(spec)
        status = self.status(spec)
        if status.state == "verified":
            self._write_sidecar(spec, status)
            return status

        for file_spec in (self._sam_download_spec(spec), self._mam_download_spec(spec)):
            if is_cancelled():
                raise RefinerDownloadCancelled("refiner download cancelled")
            self._ensure_file(spec, file_spec, progress, is_cancelled)

        verified = self.status(spec)
        if verified.state != "verified":
            raise RefinerVerificationError("MAM checkpoint verification failed")
        self._write_sidecar(spec, verified)
        return verified

    def checkpoint_path(self, spec: RefinerSpec, role: Literal["sam", "mam"]) -> Path:
        file_spec = (
            self._sam_download_spec(spec) if role == "sam" else self._mam_download_spec(spec)
        )
        return self._refiner_dir(spec) / file_spec.filename

    def partial_path(self, spec: RefinerSpec, role: Literal["sam", "mam"]) -> Path:
        checkpoint_path = self.checkpoint_path(spec, role)
        return checkpoint_path.with_name(checkpoint_path.name + ".partial")

    def _validate_spec(self, spec: RefinerSpec) -> None:
        if (
            not spec.sam_checkpoint_url
            or not spec.sam_checkpoint_filename
            or spec.sam_expected_bytes <= 0
            or not spec.mam_checkpoint_url
            or not spec.mam_checkpoint_filename
            or spec.mam_expected_bytes <= 0
        ):
            raise RefinerUnavailable("MAM checkpoint is not available")

    def _refiner_dir(self, spec: RefinerSpec) -> Path:
        return self._root / spec.id

    def _ensure_refiner_dir(self, spec: RefinerSpec) -> None:
        try:
            self._refiner_dir(spec).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RefinerCacheDirMissing("edge refiner cache is not writable") from exc
        if not os.access(self._refiner_dir(spec), os.W_OK):
            raise RefinerCacheDirMissing("edge refiner cache is not writable")

    def _sam_download_spec(self, spec: RefinerSpec) -> _DownloadSpec:
        return _DownloadSpec(
            role="sam",
            url=spec.sam_checkpoint_url,
            filename=spec.sam_checkpoint_filename,
            expected_bytes=spec.sam_expected_bytes,
            etag=spec.sam_etag,
        )

    def _mam_download_spec(self, spec: RefinerSpec) -> _DownloadSpec:
        return _DownloadSpec(
            role="mam",
            url=spec.mam_checkpoint_url,
            filename=spec.mam_checkpoint_filename,
            expected_bytes=spec.mam_expected_bytes,
            etag=spec.mam_etag,
        )

    def _file_status(
        self, spec: RefinerSpec, file_spec: _DownloadSpec
    ) -> tuple[CacheState, int | None]:
        path = self._refiner_dir(spec) / file_spec.filename
        partial = path.with_name(path.name + ".partial")
        if path.exists():
            actual_bytes = path.stat().st_size
            state: CacheState = (
                "verified" if actual_bytes == file_spec.expected_bytes else "invalid"
            )
            return state, actual_bytes
        if partial.exists():
            return "partial", partial.stat().st_size
        return "missing", None

    def _ensure_file(
        self,
        spec: RefinerSpec,
        file_spec: _DownloadSpec,
        progress: ProgressCallback,
        is_cancelled: CancelCallback,
    ) -> None:
        checkpoint_path = self._refiner_dir(spec) / file_spec.filename
        partial_path = checkpoint_path.with_name(checkpoint_path.name + ".partial")
        if checkpoint_path.exists():
            actual_bytes = checkpoint_path.stat().st_size
            if actual_bytes == file_spec.expected_bytes:
                return
            invalid_path = checkpoint_path.with_name(checkpoint_path.name + ".invalid")
            invalid_path.unlink(missing_ok=True)
            checkpoint_path.replace(invalid_path)

        start_offset = partial_path.stat().st_size if partial_path.exists() else 0
        if start_offset > file_spec.expected_bytes:
            partial_path.unlink()
            start_offset = 0
        self._ensure_disk_space(
            partial_path.parent, file_spec.filename, file_spec.expected_bytes - start_offset
        )

        log_event(
            "refiner.download.start",
            component="model_cache",
            **{
                "refiner.file.role": file_spec.role,
                "download.bytes_done": start_offset,
            },
        )
        try:
            try:
                downloaded = self._download_once(
                    file_spec, partial_path, start_offset, progress, is_cancelled
                )
            except RefinerDownloadError:
                if start_offset <= 0:
                    raise
                partial_path.unlink(missing_ok=True)
                downloaded = self._download_once(file_spec, partial_path, 0, progress, is_cancelled)
        except RefinerDownloadCancelled:
            raise
        except RefinerDownloadError as exc:
            log_event(
                "refiner.download.failed",
                severity="error",
                component="model_cache",
                **{
                    "refiner.file.role": file_spec.role,
                    "error.type": exc.__class__.__name__,
                    "error.reason": _exception_reason(exc),
                },
            )
            raise
        except (HTTPError, TimeoutError, URLError, OSError) as exc:
            log_event(
                "refiner.download.failed",
                severity="error",
                component="model_cache",
                **{
                    "refiner.file.role": file_spec.role,
                    "error.type": exc.__class__.__name__,
                    "error.reason": _exception_reason(exc),
                },
            )
            raise RefinerDownloadError("MAM checkpoint download failed") from exc

        if downloaded != file_spec.expected_bytes:
            raise RefinerVerificationError("MAM checkpoint verification failed")

        partial_path.replace(checkpoint_path)
        log_event(
            "refiner.download.result",
            component="model_cache",
            **{
                "refiner.file.role": file_spec.role,
                "model.cache.state": "verified",
                "model.checkpoint.bytes": downloaded,
            },
        )

    def _download_once(
        self,
        file_spec: _DownloadSpec,
        partial_path: Path,
        start_offset: int,
        progress: ProgressCallback,
        is_cancelled: CancelCallback,
    ) -> int:
        mode = "ab" if start_offset > 0 else "wb"
        downloaded = start_offset
        with partial_path.open(mode) as file:
            while downloaded < file_spec.expected_bytes:
                if is_cancelled():
                    raise RefinerDownloadCancelled("refiner download cancelled")
                range_end = min(downloaded + CHUNK_SIZE - 1, file_spec.expected_bytes - 1)
                chunk = self._read_range_with_retries(
                    file_spec, downloaded, range_end, is_cancelled
                )
                file.write(chunk)
                downloaded += len(chunk)
                progress_value = min(downloaded / file_spec.expected_bytes, 1.0)
                progress(
                    "downloading_refiner",
                    progress_value,
                    {
                        "bytes_done": downloaded,
                        "bytes_total": file_spec.expected_bytes,
                    },
                )
        return downloaded

    def _read_range_with_retries(
        self,
        file_spec: _DownloadSpec,
        range_start: int,
        range_end: int,
        is_cancelled: CancelCallback,
    ) -> bytes:
        expected_length = range_end - range_start + 1
        last_error: Exception | None = None
        for attempt in range(1, DOWNLOAD_RETRY_ATTEMPTS + 1):
            if is_cancelled():
                raise RefinerDownloadCancelled("refiner download cancelled")
            request = Request(file_spec.url)
            request.add_header("Range", f"bytes={range_start}-{range_end}")
            request.add_header("User-Agent", "Mozilla/5.0")
            try:
                with self._opener(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                    if response.headers.get("Content-Range") is None:
                        raise RefinerDownloadError("MAM checkpoint download failed")
                    chunk = response.read(expected_length)
                    if len(chunk) != expected_length:
                        raise RefinerDownloadError("MAM checkpoint download failed")
                    return chunk
            except RefinerDownloadError as exc:
                last_error = exc
                break
            except (HTTPError, TimeoutError, URLError, OSError) as exc:
                last_error = exc
                if attempt == DOWNLOAD_RETRY_ATTEMPTS:
                    break
                log_event(
                    "refiner.download.retry",
                    severity="warning",
                    component="model_cache",
                    **{
                        "refiner.file.role": file_spec.role,
                        "retry.attempt": attempt,
                        "error.type": exc.__class__.__name__,
                        "error.reason": _exception_reason(exc),
                        "download.range_start": range_start,
                        "download.range_end": range_end,
                    },
                )
        raise RefinerDownloadError("MAM checkpoint download failed") from last_error

    def _ensure_disk_space(self, directory: Path, filename: str, required_bytes: int) -> None:
        free_bytes = shutil.disk_usage(directory).free
        if free_bytes < required_bytes:
            log_event(
                "refiner.download.failed",
                severity="error",
                component="model_cache",
                **{
                    "file.name": filename,
                    "error.type": "InsufficientDiskSpace",
                    "error.reason": "insufficient disk space",
                    "disk.bytes_required": required_bytes,
                    "disk.bytes_free": free_bytes,
                },
            )
            raise RefinerDownloadError("MAM checkpoint download failed")

    def _write_sidecar(self, spec: RefinerSpec, status: RefinerCacheStatus) -> None:
        sidecar_path = self._refiner_dir(spec) / "refiner.json"
        payload = {
            "id": spec.id,
            "title": spec.title,
            "sam_checkpoint_filename": spec.sam_checkpoint_filename,
            "sam_expected_bytes": spec.sam_expected_bytes,
            "sam_actual_bytes": status.sam_actual_bytes,
            "sam_etag": spec.sam_etag,
            "mam_checkpoint_filename": spec.mam_checkpoint_filename,
            "mam_expected_bytes": spec.mam_expected_bytes,
            "mam_actual_bytes": status.mam_actual_bytes,
            "mam_etag": spec.mam_etag,
            "cache_state": status.state,
        }
        sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
