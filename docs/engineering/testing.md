# Testing

## Principles

Тесты должны фокусироваться на поведении, которое можно проверять вне Krita.

Приоритет:

- worker protocol;
- model registry validation;
- download/cache logic;
- hit-testing masks;
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
3. Confirm preparing/ready state in Tool Options.
4. Hover fake segments.
5. Confirm red overlay.
6. Click hovered segment.
7. Confirm Krita selection.
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

## Current Covered Cases

Python worker tests cover:

- fake segmentation creates 3 segments;
- segment metadata is produced;
- generated segment map is hit-testable;
- `prepare_segments` emits progress events;
- worker writes valid `.npy <u2`;
- `.npy` shape is `[height, width]`;
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
- `exactSnapshotFingerprint()` is stable for equal images;
- fingerprint changes when pixels change;
- fingerprint changes when image size changes;
- null image fingerprint is empty.

## Future Test Additions

- Typed worker parser DTO tests.
- Snapshot export tests for logic separable from Krita.
- Cache key invalidation tests.
- ModelRegistry validation tests.
- Model download/cache tests.
- SAM2 result conversion tests.
- Worker crash/error tests.
- Selection modifier mapping tests.
- Coordinate mapping tests around zoom/pan where feasible.
