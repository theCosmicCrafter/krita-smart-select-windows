# Snapshot And Cache

## Input Image Rules

MVP сегментирует flattened projection текущего документа по размеру canvas.

Rules:

- Входом является изображение целиком, а не активный слой.
- Используется flattened projection всех видимых слоёв.
- Невидимые слои не учитываются.
- Bounds равны размеру canvas.
- Прозрачность считается фоном, а не отдельным объектом.
- Для RGB-входа модели прозрачные пиксели композитятся на стабильный белый фон.
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
- fingerprint изображения.

## Exact Fingerprint

MVP fingerprint должен быть консервативным, но в текущей реализации это
downscaled fingerprint, а не hash всех пикселей:

1. Получить snapshot всего изображения.
2. Композитить RGB input на белый фон.
3. Сохранить alpha/background mask.
4. Downscale RGB input и alpha mask до `128x128`.
5. Привести к stable RGB/alpha U8 representation.
6. Посчитать hash bytes.

Для MVP используется exact match по fingerprint string. Так как fingerprint
downscaled, он не гарантирует обнаружение каждого single-pixel изменения в
полном размере изображения.

## Reuse Rules

- Если размер изображения изменился, кэш невалиден.
- Если model id изменился, кэш невалиден.
- Если algorithm version изменилась, кэш невалиден.
- Если fingerprint совпал, кэш можно использовать.
- Если fingerprint не совпал, кэш невалиден.
- Если есть сомнение, кэш нужно инвалидировать и пересчитать сегменты.

Runtime rebuild при активном Smart Select debounce-ится на `300 ms`, чтобы
частые `sigImageUpdated` события не перезапускали worker на каждый update.

Current real inference algorithm version:

```text
sam2.1-auto-mask-v1
```

Эта версия заменила `mock-segments-model-readiness-v1` и намеренно
инвалидирует in-memory cache, созданный readiness/mock stage.

## Follow-Ups

После MVP можно добавить:

- disk cache;
- perceptual hash;
- mean absolute difference;
- Jaccard/IoU по карте изменённых пикселей;
- similarity thresholds для reuse при небольших изменениях.

В MVP similarity thresholds не используются.
