# Testing

## Default Check

```bash
PYTHON=.venv/bin/python ./scripts/test-code.sh
```

This runs:

- Python compile checks;
- Ruff format check;
- Ruff lint;
- mypy strict;
- Python worker unit tests;
- C++ core tests when Krita source/build paths are available.

## Python Coverage

Current unit tests cover:

- BiRefNet model registry metadata;
- BiRefNet model cache verification and pinned snapshot download call shape;
- runtime status/rebuild behavior;
- runtime diagnostics without installing dependencies;
- BiRefNet inference wrapper with fake torch/transformers/PIL modules;
- alpha selection builder clipping and soft alpha preservation;
- region/lasso helper behavior;
- worker `prepare_selection` success/error/cancel paths with fake inference;
- OTel-compatible worker log formatting.

Default tests must not download the real model or install torch.

## C++ Coverage

Core tests cover:

- snapshot fingerprint and alpha inclusion;
- snapshot normalization;
- `.npy` writer;
- `PrepareSelectionRequest` serialization;
- soft `SelectionMask` loading and alpha preservation;
- lasso region mask rasterization.

## Opt-In Real BiRefNet Smoke

The real BiRefNet smoke is intentionally explicit because it may create the
managed runtime and download the model snapshot:

```bash
SMART_SELECT_RUN_BIREFNET_SMOKE=1 ./scripts/test-birefnet-smoke.sh
```

It is not called by `scripts/test-code.sh`.

## Manual GUI Smoke

1. Build and install dist.
2. Start Krita AppImage.
3. Open a small image.
4. Select Smart Select.
5. Confirm no inference starts on activation.
6. Draw lasso around a subject.
7. Edit the selected region with add/subtract/intersect actions.
8. Press `Select object with AI`.
9. Confirm Tool Options shows alpha runtime/model stages.
10. Confirm result is applied automatically.
11. Confirm selection stays inside the selected region.
12. Confirm transparent document pixels are excluded.
13. Confirm click outside current selection deselects when native Krita behavior allows it.
14. Confirm cancel during setup does not crash.
15. Confirm late inference result is ignored after cancel/tool switch.
16. Confirm logs include `selection.start`, `alpha_model.predict.result`,
    `alpha_mask.result` and `selection.result`.
