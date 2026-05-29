# Product Scope

## Рабочее Название

Название плагина: **Krita Smart Select**.

В интерфейсе Krita можно использовать короткие подписи:

- Smart Select
- Smart Select Settings

## Цель

Krita Smart Select помогает создавать выделения в Krita с помощью AI.

Ожидаемый опыт должен быть ближе к нативному инструменту выделения, чем к
отдельному окну с миниатюрой изображения. Пользователь работает на настоящем
canvas Krita: грубо задаёт область интереса обычным жестом выделения, а SAM2
уточняет её до объекта или части объекта и применяет результат как обычное
выделение Krita.

## Основной UX

1. Пользователь активирует Smart Select tool.
2. Пользователь обводит область интереса lasso/freehand selection gesture или
   задаёт прямоугольную область.
3. Плагин экспортирует snapshot/crop этой области с небольшим padding и
   передаёт worker-у region prompt.
4. Worker запускает SAM2.1 Large на ограниченной области и строит
   recall-oriented SAM prior.
5. Worker уточняет prior через MAM SAM ViT-H и получает soft alpha mask.
6. Пока сегментация выполняется, Krita остаётся отзывчивой, а tool показывает
   progress/loader в canvas/tool options.
7. Tool применяет mask как выделение Krita для завершённой lasso-области.
8. Клавиши-модификаторы должны следовать нативному поведению selection tools в
   Krita.

## Region-Prompted Segmentation Model

Первая версия переходит на quality-first region-prompted workflow:

- пользователь сначала ограничивает область поиска;
- SAM2 запускается на crop/region prompt вместо всего изображения;
- завершение lasso gesture применяет уточнённую mask как Krita selection;
- автоматическая full-image segment map больше не является центральным
  сценарием MVP.

Предыдущий prepared full-image hover/click flow остаётся полезным canvas spike и
может вернуться как отдельный быстрый режим, но качество MVP теперь оценивается
по region-prompted результату.

## MVP Scope

В MVP входит:

- flattened projection всех видимых слоёв;
- lasso/freehand region input на canvas;
- SAM2.1 Large segmentation по ограниченной области;
- MAM-H alpha refinement поверх SAM prior;
- soft alpha selection mask;
- apply-to-selection;
- selection modifiers по нативным conventions Krita;
- worker-backed inference pipeline;
- in-memory runtime cache.

Отдельная команда `Generate all masks for image/layer` в MVP не нужна. Full-image
automatic masks считаются отдельным режимом/follow-up, а не обязательным шагом
при активации Smart Select.

## Future Commands

После стабилизации MVP можно добавить:

- `Refresh segments` / `Rebuild smart select cache` для ручного пересчёта.
- `Extract selection to new layer` для создания нового слоя из выбранного
  сегмента.
- Full-image prepared hover/click mode для быстрых выборов без lasso.
- Candidate mask debug export для анализа качества SAM2 и постобработки.
- Preview/confirm flow для refined alpha перед применением.
- Click-to-select сегменты из prepared full-image карты.

## Non-Goals Первой Версии

- Загрузка пользовательских кастомных моделей.
- Поддержка GGUF-моделей.
- SAM3 как default или обязательный backend.
- Отдельный режим `sample current layer`.
- Сложное generative editing или inpainting.
