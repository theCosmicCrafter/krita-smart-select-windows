# Snapshot And Cache

## Input Image Rules

MVP сегментирует flattened projection текущего документа по размеру canvas.

Rules:

- Входом является изображение целиком, а не активный слой.
- Используется flattened projection всех видимых слоёв.
- Невидимые слои не учитываются.
- Bounds равны размеру canvas.
- Прозрачность считается фоном, а не отдельным объектом.
- Для RGB-входа модели прозрачные пиксели нужно композитить на стабильный фон.
- Alpha/background mask нужно сохранить, чтобы полностью прозрачные области не
  предлагались как selectable segments.

Режимы `sample current layer` и segmentation only active selection являются
future work.

## Cache Scope

Кэш сегментов относится к изображению целиком.

MVP не пытается отдельно кэшировать:

- активный слой;
- активное выделение;
- prompt-based requests;
- отдельные layer modes.

Segment map, masks и metadata в MVP хранятся в памяти. Дисковый кэш можно
добавить позже, если подготовка сегментов окажется достаточно дорогой для
частого пересчёта между сессиями.

## Cache Key

Ключ кэша должен включать:

- стабильный id документа, если его можно получить или сохранить;
- размер изображения;
- цветовую модель/depth, если доступно;
- registry id выбранной модели;
- версию алгоритма подготовки сегментов;
- exact fingerprint изображения.

## Exact Fingerprint

MVP fingerprint должен быть консервативным:

1. Получить snapshot всего изображения.
2. Downscale snapshot до `64x64` или `128x128`.
3. Привести к stable RGB/RGBA U8 representation.
4. Посчитать hash bytes.

Для MVP используется exact match only.

## Reuse Rules

- Если размер изображения изменился, кэш невалиден.
- Если model id изменился, кэш невалиден.
- Если algorithm version изменилась, кэш невалиден.
- Если fingerprint совпал, кэш можно использовать.
- Если fingerprint не совпал, кэш невалиден.
- Если есть сомнение, кэш нужно инвалидировать и пересчитать сегменты.

## Follow-Ups

После MVP можно добавить:

- disk cache;
- perceptual hash;
- mean absolute difference;
- Jaccard/IoU по карте изменённых пикселей;
- similarity thresholds для reuse при небольших изменениях.

В MVP similarity thresholds не используются.
