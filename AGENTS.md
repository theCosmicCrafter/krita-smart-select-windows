# AGENTS.md

Короткая памятка для агентов, работающих над Krita Smart Select.

## Сначала Читать

- `docs/README.md` - карта документации.
- `DESIGN.md` - короткий индекс, не основной дизайн-документ.
- `docs/architecture.md` - C++/Python split и canvas tool rules.
- `docs/worker-protocol.md` - NDJSON protocol и segment map contract.
- `docs/runtime-env.md` - managed SAM2 venv/cache policy.
- `docs/engineering/typing.md` - mypy/stubs policy.
- `docs/engineering/testing.md` - текущие тесты и команды.
- `docs/engineering/observability.md` - logging/diagnostics/support rules.

## Базовые Решения

- MVP target: Linux + Krita 5.3.1.
- Canvas/tool layer: native C++ Krita tool.
- Python layer: loader wrapper and worker process.
- Worker transport: `QProcess` + stdio NDJSON.
- Large data transport: temporary files, not JSON.
- Segment map contract: NumPy `.npy`, dtype `<u2`, shape `[height, width]`,
  value `0` = background/no segment.
- Lasso selection mask contract: NumPy `.npy`, dtype `|u1`, shape
  `[height, width]`, values `0..255` are soft selection alpha and must not be
  binarized in C++.
- Runtime cache for MVP: in-memory only.
- Current default real model: `sam2.1_hiera_large`.
- Real model target: SAM2/SAM2.1 PyTorch, curated models only.
- Current MVP inference path uses SAM2.1 Large lasso/region-prompted selection.
- Current refinement path uses MAM SAM ViT-H (`mam_sam_vith`) on top of a
  trimap generated from the recall-oriented SAM2 prior. MAM may refine only the
  trimap unknown band; the eroded core stays definite foreground, and output is
  rejected if it grows or shrinks too far from the SAM prior.
- Worker builds a recall-oriented SAM prior from multiple prompt variants and
  compatible candidates; do not treat one highest-score SAM mask as final when
  improving quality.
- Current MVP product direction is lasso/region-prompted selection, not
  click-to-select from full-image prepared masks.
- Lasso completion should preserve native Krita selection behavior, including
  click-outside deselect through Krita selection helper semantics.
- Runtime venv/cache is managed automatically under Krita user data, not inside
  plugin source.
- Current algorithm version: `sam2.1-lasso-trimap-mam-v1`.

## Не Переоткрывать Без Причины

- Do not implement canvas overlay as a floating QWidget or Python-only overlay.
- Do not pass masks/pixel arrays through JSON.
- Do not add GGUF/custom models/SAM3 to MVP scope.
- Do not switch the default model back to Tiny/Small unless the product decision
  is explicitly reopened.
- Do not install PyTorch/SAM2 into the plugin source tree.
- Do not reintroduce click-to-select full-image hover as the primary MVP path
  unless the product decision is explicitly reopened.
- Do not silently fall back from MAM ViT-H to MAM ViT-B in the product path.
- Do not binarize soft alpha selection masks before Krita selection apply.
- Do not add GUI automation as an MVP requirement.

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
- Logs must be OpenTelemetry-compatible by schema, but the MVP must not add an
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

Stable decisions belong in `docs/`, not in long comments or a revived
`DESIGN.md`.

When changing architecture, protocol, typing, testing or support behavior,
update the relevant doc in the same change.

## Commits

Commit messages should conform to the conventional commits standard
