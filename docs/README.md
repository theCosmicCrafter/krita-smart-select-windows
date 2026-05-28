# Krita Smart Select Documentation

Krita Smart Select - плагин для Krita, который помогает создавать выделения с
помощью AI. Основной сценарий: пользователь активирует Smart Select, плагин
готовит сегменты текущего изображения, затем пользователь выбирает подходящий
сегмент прямо на canvas через hover/click.

## Статус MVP

- Target platform: Linux.
- Target Krita version: Krita 5.3.1.
- UI/canvas layer: native C++ Krita tool.
- Python layer: loader wrapper and worker process.
- Worker transport: `QProcess` + stdio NDJSON.
- Large data transport: temporary files.
- Current implementation: canvas spike, worker-backed mock segmentation,
  flattened snapshot export, alpha mask handling and in-memory cache reuse.
- Not implemented yet: SAM2 inference.

## Где Что Искать

- [Product Scope](product-scope.md): UX, MVP scope, non-goals.
- [Architecture](architecture.md): runtime split, canvas tool, packaging target.
- [Worker Protocol](worker-protocol.md): NDJSON protocol, temp files, segment map contract.
- [Snapshot And Cache](snapshot-cache.md): input image, exact fingerprint, cache rules.
- [Models](models.md): SAM2 strategy, ModelRegistry, model download/cache.
- [Observability](engineering/observability.md): logs, diagnostic report, support flow.
- [Typing](engineering/typing.md): mypy, local stubs, typed boundaries.
- [Testing](engineering/testing.md): test layers, current coverage, commands.
- [References And Licensing](references-and-licensing.md): reference projects and license direction.

## Documentation Rule

`DESIGN.md` is only an index. Stable decisions belong in these `docs/` files.
