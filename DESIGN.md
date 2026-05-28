# Krita Smart Select Design Index

## Статус

Полноценная документация перенесена в `docs/`. Этот файл остаётся коротким
entrypoint и указателем на актуальные документы.

Проект: **Krita Smart Select**.

Текущий MVP target:

- Linux.
- Krita 5.3.1.
- Native C++ canvas tool + Python loader/worker.
- Worker transport: `QProcess` + stdio NDJSON.
- Основной UX direction: lasso/region-prompted Smart Select, без
  click-to-select по full-image prepared сегментам.

## Документация

- [Documentation Map](docs/README.md)
- [Product Scope](docs/product-scope.md)
- [Architecture](docs/architecture.md)
- [Worker Protocol](docs/worker-protocol.md)
- [Snapshot And Cache](docs/snapshot-cache.md)
- [Models](docs/models.md)
- [Observability](docs/engineering/observability.md)
- [Typing](docs/engineering/typing.md)
- [Testing](docs/engineering/testing.md)
- [References And Licensing](docs/references-and-licensing.md)

## Правило Обновления

Новые стабильные решения нужно записывать сразу в соответствующий файл в
`docs/`. Этот файл не должен снова становиться полной копией дизайн-документа.
