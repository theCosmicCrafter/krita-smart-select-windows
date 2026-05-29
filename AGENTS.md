# AGENTS.md

Короткая памятка для агентов, работающих над Krita Smart Select.

## Сначала Читать

- `README.md` - установка, запуск, dev setup и карта документации.
- `docs/architecture.md` - C++/Python split и canvas tool rules.
- `docs/worker-protocol.md` - NDJSON protocol и selection mask contract.
- `docs/runtime-env.md` - managed BiRefNet venv/cache policy.
- `docs/engineering/typing.md` - mypy/stubs policy.
- `docs/engineering/testing.md` - текущие тесты и команды.
- `docs/engineering/code-style.md` - Ruff, clang-format и стиль кода.
- `docs/engineering/observability.md` - logging/diagnostics/support rules.
- `docs/licensing.md` - лицензирование кода, модели и релизных пакетов.

## Базовые Решения

- Target: Linux + Krita 5.3.1.
- Canvas/tool layer: native C++ Krita tool.
- Python layer: loader wrapper and worker process.
- Worker transport: `QProcess` + stdio NDJSON.
- Large data transport: temporary files, not JSON.
- Current primary flow: lasso-driven alpha selection with explicit `Select
  object with AI` action.
- Current model: `birefnet_hr_matting` (`ZhengPeng7/BiRefNet_HR-matting`).
- Current algorithm version: `birefnet-hr-matting-lasso-v1`.
- BiRefNet is used as a direct alpha extraction model.
- `prepare_selection` is the only inference protocol path.
- Lasso selection mask contract: NumPy `.npy`, dtype `|u1`, shape
  `[height, width]`, values `0..255` are soft selection alpha and must not be
  binarized in C++.
- Worker clips model output to `selection_region & document_alpha`.
- Lasso completion edits the native Krita selection first; BiRefNet inference
  starts only when the user presses `Select object with AI`, using the current
  selection as the region mask.
- Runtime venv/cache is managed automatically under Krita user data, not inside
  plugin source.
- The worker keeps disk caches, but unloads the CUDA BiRefNet model and clears
  PyTorch CUDA allocator cache after each lasso job to avoid VRAM buildup in
  Krita.
- `trust_remote_code=True` is allowed only for the pinned curated BiRefNet repo
  and revision recorded in `docs/models.md`.
- Lasso completion should preserve native Krita selection behavior, including
  click-outside deselect through Krita selection helper semantics.
- Debug exports may include `crop_rgb.png`, `crop_region_mask.png`,
  `crop_alpha_mask.png`, `birefnet_alpha_raw.png`, `final_alpha.png` and
  `metadata.json`.

## Не Переоткрывать Без Причины

- Do not implement canvas overlay as a floating QWidget or Python-only overlay.
- Do not pass masks/pixel arrays through JSON.
- Do not install PyTorch or model dependencies into the plugin source tree.
- Do not binarize soft alpha selection masks before Krita selection apply.
- Do not add hidden final grow; Grow/Feather/Anti-aliasing remain native Krita
  post-selection options.

## Typing And Tests

Use the existing typed-islands approach:

- Strict Python typing for `worker/` and `python/smart_select/`.
- Local Krita stubs live in `typings/`.
- Keep Krita/PyQt adapter typing pragmatic.
- Python formatting/linting uses Ruff.
- C++ formatting uses `.clang-format`; do not mass-format unrelated Krita code.

Recommended check:

```bash
PYTHON=.venv/bin/python ./scripts/test-code.sh
```

If `.venv` is missing:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
PYTHON=.venv/bin/python ./scripts/test-code.sh
```

For native plugin build:

```bash
BUILD_JOBS=20 ./scripts/build-krita-5.3.1.sh
```

## Observability Rules

- Use stable structured event names.
- Logs must be OpenTelemetry-compatible by schema, but the plugin must not add an
  OpenTelemetry SDK, OTLP exporter or network telemetry without an explicit
  product decision.
- Use `event.name`, not `event`, for structured log records.
- Prefer dotted field names such as `job.id`, `job.type`, `model.id`,
  `error.type`, `device.type`, `duration_ms` and `selection.area`.
- Log worker lifecycle, inference lifecycle, cancellation, errors and cleanup.
- Do not log image contents, pixel data, masks, tokens or secrets.
- Redact home paths and document paths in user-facing diagnostic reports where
  possible.

## Documentation Rule

Stable decisions belong in `docs/`, not in long comments.

When changing architecture, protocol, typing, testing or support behavior,
update the relevant doc in the same change.

## Commits

Commit messages should conform to the conventional commits standard.
