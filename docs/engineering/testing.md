# Testing

## Principles

Тесты должны фокусироваться на поведении, которое можно проверять вне Krita.

Приоритет:

- worker protocol;
- model registry validation;
- download/cache logic;
- lasso/region mask rasterization;
- selection mask generation;
- coordinate mapping;
- mask combining;
- error handling and cancellation;
- image fingerprint/cache invalidation.

Krita GUI integration проверяется manual smoke в MVP. Автоматизированный GUI
smoke не обязателен для MVP, потому что coordinate-driven UI automation для
Krita будет хрупким и дорогим.

## Test Layers

### Python

- `py_compile` for syntax/import sanity.
- `mypy` for typed Python boundaries.
- `unittest` for worker protocol and generated file contracts.

### C++

- QtTest for pure core modules.
- Current target: `smartselect_core_tests`.

### Manual GUI Smoke

Manual smoke остаётся нужным для canvas/tool поведения:

1. Enable plugin.
2. Activate Smart Select.
3. Draw a lasso/freehand region around an object or object part.
4. Confirm preparing state in Tool Options.
5. Confirm result preview appears on canvas.
6. Confirm Krita selection is applied from the lasso-refined mask.
7. Repeat with selection modifiers.
8. Switch tool and reselect Smart Select.
9. Close Krita without crash.

## Current Command

```bash
PYTHON=.venv/bin/python ./scripts/test-code.sh
```

The script runs:

1. Python compile checks.
2. Ruff format check.
3. Ruff lint check.
4. `mypy`.
5. Python worker unit tests.
6. C++ QtTest target build.
7. CTest for `smartselect_core_tests`.

## Opt-In Real SAM2 Smoke

The real SAM2 smoke is intentionally not part of the default suite because it
may install PyTorch/SAM2 and download the SAM2.1 Large checkpoint:

```bash
SMART_SELECT_RUN_SAM2_SMOKE=1 ./scripts/test-sam2-smoke.sh
```

It creates a tiny synthetic snapshot, uses the normal worker protocol, writes
runtime/model caches under Krita user data by default and verifies that the
worker returns a non-empty selection mask.

## Current Covered Cases

Python worker tests cover:

- model registry default is `sam2.1_hiera_large`;
- SAM2.1 Large registry metadata is stable;
- model cache reuses verified files without network calls;
- model cache downloads through `.partial` and writes `model.json`;
- model cache resumes partial downloads when ranges are supported;
- model cache restarts when resume is not supported;
- model cache preserves partial files on cancellation;
- model download failures log stable error type and short reason to stderr;
- runtime diagnostics succeeds without torch installed;
- runtime diagnostics reports model cache state;
- runtime status detects ready and invalid runtime markers;
- runtime bootstrap rejects plugin-source cache paths;
- SAM2 inference wrapper chooses CUDA when available;
- SAM2 inference wrapper falls back to CPU only when CUDA is unavailable;
- CUDA OOM maps to `InferenceOutOfMemory`;
- SAM2 generator is cached inside the worker process;
- SAM2 predictor calls `set_image()` before `predict()`;
- SAM2 predictor uses `multimask_output=True`;
- image preprocessing caps inference long side at `2048`;
- lasso region masks load as `uint8`;
- lasso region bbox and prompt points are computed;
- lasso selection masks are clipped to region and alpha;
- low-overlap SAM candidates fall back to the lasso region;
- SAM mask conversion excludes transparent alpha pixels;
- SAM mask conversion resolves overlaps by score/stability/area priority;
- tiny masks are dropped from segment maps;
- fake segmentation creates 3 segments;
- segment metadata is produced;
- generated segment map is hit-testable;
- `prepare_segments` emits progress events;
- real model with fake SAM emits runtime/model/inference stages;
- mock `prepare_selection` returns a valid `selection_mask_path`;
- missing real model runtime dir is rejected;
- unknown model id is rejected;
- `diagnostics` request returns runtime info;
- worker writes valid `.npy <u2`;
- `.npy` shape is `[height, width]`;
- worker reads valid `uint8` alpha masks;
- worker rejects invalid alpha masks;
- transparent alpha mask pixels are excluded from mock segment maps;
- metadata is recomputed after alpha masking;
- invalid canvas size is rejected;
- non-object request params are rejected;
- cancel request records cancelled job;
- worker process accepts NDJSON request through stdin and returns result through
  stdout.

C++ tests cover:

- `SegmentMap::loadNpy()` accepts valid uint16 `.npy`;
- wrong dtype is rejected;
- Fortran-order arrays are rejected;
- truncated payload is rejected;
- `segmentAt()` returns expected ids and handles out-of-bounds as background;
- segment bounds are tight;
- `selectionMask()` returns correct bounds and alpha bytes;
- `exactSnapshotFingerprint()` is stable for equal RGB/alpha input;
- fingerprint changes when RGB pixels change;
- fingerprint changes when alpha mask changes;
- fingerprint changes when image size changes;
- invalid alpha mask length is rejected;
- null image fingerprint is empty;
- snapshot normalization composites transparent pixels over white;
- snapshot normalization preserves alpha mask bytes;
- fully transparent snapshots are detected;
- `writeUint8Npy()` writes valid NumPy headers;
- `writeUint8Npy()` rejects wrong payload sizes.
- `prepare_segments` request serialization includes `model_cache_dir` and
  `runtime_cache_dir`.
- `prepare_selection` request serialization includes lasso region fields and
  runtime/model cache dirs;
- `SelectionMask::loadNpy()` accepts valid `uint8` `.npy`;
- lasso rasterization creates a non-empty bbox and mask bytes.

## Future Test Additions

- Cache key invalidation tests.
- Worker crash/error tests.
- Selection modifier mapping tests.
- Lasso region mask serialization tests.
- `prepare_selection` request/result tests.
- Coordinate mapping tests around zoom/pan where feasible.
