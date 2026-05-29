# Krita Smart Select Documentation

Krita Smart Select - плагин для Krita, который помогает создавать выделения с
помощью AI. Основной сценарий MVP: пользователь активирует Smart Select,
обводит объект лассо, worker запускает BiRefNet HR Matting на crop вокруг
области и возвращает soft alpha selection mask.

## Статус MVP

- Target platform: Linux.
- Target Krita version: Krita 5.3.1.
- UI/canvas layer: native C++ Krita tool.
- Python layer: loader wrapper and worker process.
- Worker transport: `QProcess` + stdio NDJSON.
- Large data transport: temporary files.
- Current implementation: native canvas flow, lasso input, flattened snapshot
  export, document alpha mask, lasso region mask, managed BiRefNet runtime,
  pinned Hugging Face model cache and soft alpha selection apply.
- Current product direction: plug-and-play lasso alpha extraction. SAM2/MAM,
  trimap tuning, prepared hover segmentation and click-to-select are out of the
  current MVP path.
- Not implemented yet: model picker, alternative models, persistent logs and
  support dialog.

## Где Что Искать

- [Product Scope](product-scope.md): UX, MVP scope, non-goals.
- [Architecture](architecture.md): runtime split, canvas tool, packaging target.
- [Worker Protocol](worker-protocol.md): NDJSON protocol and selection mask contract.
- [Snapshot And Cache](snapshot-cache.md): input image, exact fingerprint, cache rules.
- [Models](models.md): BiRefNet model metadata, download/cache and follow-ups.
- [Runtime Environment](runtime-env.md): managed venv/cache for BiRefNet runtime.
- [Observability](engineering/observability.md): logs, diagnostic report, support flow.
- [Typing](engineering/typing.md): mypy, local stubs, typed boundaries.
- [Testing](engineering/testing.md): test layers, current coverage, commands.
- [References And Licensing](references-and-licensing.md): reference projects and license direction.

## Documentation Rule

`DESIGN.md` is only an index. Stable decisions belong in these `docs/` files.
