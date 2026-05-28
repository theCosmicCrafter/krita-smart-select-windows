# AGENTS.md

Короткая памятка для агентов, работающих над Krita Smart Select.

## Сначала Читать

- `docs/README.md` - карта документации.
- `DESIGN.md` - короткий индекс, не основной дизайн-документ.
- `docs/architecture.md` - C++/Python split и canvas tool rules.
- `docs/worker-protocol.md` - NDJSON protocol и segment map contract.
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
- Runtime cache for MVP: in-memory only.
- Current default real model: `sam2.1_hiera_large`.
- Real model target: SAM2/SAM2.1 PyTorch, curated models only.
- Current worker still returns mock segment maps after model readiness;
  real SAM2 inference is a later stage.

## Не Переоткрывать Без Причины

- Do not implement canvas overlay as a floating QWidget or Python-only overlay.
- Do not pass masks/pixel arrays through JSON.
- Do not add GGUF/custom models/SAM3 to MVP scope.
- Do not switch the default model back to Tiny/Small unless the product decision
  is explicitly reopened.
- Do not add prompt-based selection before the prepared segmentation workflow is
  stable.
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
