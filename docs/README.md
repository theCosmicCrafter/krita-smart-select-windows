# Krita Smart Select Documentation

Krita Smart Select - плагин для Krita, который помогает создавать выделения с
помощью AI. Основной сценарий MVP: пользователь активирует Smart Select,
обводит область интереса на canvas, а SAM2 уточняет эту область до объекта или
части объекта и применяет результат как обычное выделение Krita.

## Статус MVP

- Target platform: Linux.
- Target Krita version: Krita 5.3.1.
- UI/canvas layer: native C++ Krita tool.
- Python layer: loader wrapper and worker process.
- Worker transport: `QProcess` + stdio NDJSON.
- Large data transport: temporary files.
- Current implementation: native canvas flow, lasso input, flattened snapshot
  export, alpha mask handling, SAM2.1 Large registry/download/cache
  verification, automatic runtime bootstrap and real SAM2.1 Large prompted
  selection in the worker, with MAM SAM ViT-H soft alpha refinement.
- Current product direction: lasso/region-prompted selection is the primary MVP
  path; full-image prepared hover segmentation is legacy/dev compatibility.
- Not implemented yet: model picker, lighter presets, persistent logs and
  support dialog.

## Где Что Искать

- [Product Scope](product-scope.md): UX, MVP scope, non-goals.
- [Architecture](architecture.md): runtime split, canvas tool, packaging target.
- [Worker Protocol](worker-protocol.md): NDJSON protocol, temp files, segment map contract.
- [Snapshot And Cache](snapshot-cache.md): input image, exact fingerprint, cache rules.
- [Models](models.md): SAM2 strategy, ModelRegistry, model download/cache.
- [Runtime Environment](runtime-env.md): managed venv/cache for PyTorch/SAM2.
- [Observability](engineering/observability.md): logs, diagnostic report, support flow.
- [Typing](engineering/typing.md): mypy, local stubs, typed boundaries.
- [Testing](engineering/testing.md): test layers, current coverage, commands.
- [References And Licensing](references-and-licensing.md): reference projects and license direction.

## Documentation Rule

`DESIGN.md` is only an index. Stable decisions belong in these `docs/` files.
