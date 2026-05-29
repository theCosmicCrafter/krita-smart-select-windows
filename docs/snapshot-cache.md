# Snapshot And Cache

## Input Image Rules

MVP использует flattened projection текущего документа как источник пикселей,
но основной quality path сегментирует lasso/region crop, а не обязательно всё
изображение целиком.

Rules:

- Входом является flattened image snapshot и region mask, заданная lasso.
- Используется flattened projection всех видимых слоёв.
- Невидимые слои не учитываются.
- Bounds равны размеру canvas.
- Прозрачность считается фоном, а не отдельным объектом.
- Для RGB-входа модели прозрачные пиксели композитятся на стабильный белый фон.
- Alpha/background mask нужно сохранить, чтобы полностью прозрачные области не
  предлагались как selectable segments.

Режим `sample current layer` является future work.

## Current Cache Scope

Current lasso MVP не переиспользует selection results по snapshot fingerprint.
Каждое завершённое lasso запускает новый `prepare_selection` job. Fingerprint
снимка остаётся частью worker protocol и logs, но в текущем C++ path не
используется для сравнения с прошлым изображением.

MVP не пытается отдельно кэшировать:

- активный слой;
- отдельные layer modes.

Runtime/model cache остаются активными: worker переиспользует managed SAM2
runtime и verified checkpoint. Selection masks, region masks и snapshot temp
files не кэшируются между lasso jobs.

## Future Selection Cache Key

Если позже будет добавлен result cache для lasso selection, ключ должен
включать:

- стабильный id документа, если его можно получить или сохранить;
- размер изображения;
- цветовую модель/depth, если доступно;
- registry id выбранной модели;
- версию алгоритма подготовки сегментов;
- fingerprint изображения;
- fingerprint/geometry lasso region mask.

## Exact Fingerprint

Текущая реализация считает downscaled fingerprint snapshot-а, но не использует
его для reuse selection result. Fingerprint нужен как protocol/debug field и
как future input для selection cache:

1. Получить snapshot всего изображения.
2. Композитить RGB input на белый фон.
3. Сохранить alpha/background mask.
4. Downscale RGB input и alpha mask до `128x128`.
5. Привести к stable RGB/alpha U8 representation.
6. Посчитать hash bytes.

Legacy full-image real inference algorithm version:

```text
sam2.1-auto-mask-quality-v1
```

Current lasso MVP algorithm version:

```text
sam2.1-lasso-trimap-mam-v1
```

Переход на trimap-constrained MAM alpha refinement намеренно инвалидирует
assumptions предыдущего hard-mask/MAM lasso stage. Runtime/model/refiner
checkpoint caches при этом остаются переиспользуемыми.

## Follow-Ups

После MVP можно добавить:

- lasso selection result cache;
- disk cache;
- perceptual hash;
- mean absolute difference;
- Jaccard/IoU по карте изменённых пикселей;
- similarity thresholds для reuse при небольших изменениях.

В MVP similarity thresholds не используются.
