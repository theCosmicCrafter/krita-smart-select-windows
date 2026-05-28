# Architecture

## Runtime Principles

Inference не должен блокировать UI Krita.

Предпочтительная архитектура:

- UI плагина и canvas integration остаются отзывчивыми.
- Тяжёлая загрузка моделей и сегментация выполняются в worker-процессе.
- Worker отправляет progress, cancellation, result и error events.
- Сбои PyTorch, CUDA или модельного кода не должны крашить или замораживать
  Krita.

## C++ / Python Split

C++ часть отвечает за:

- native Krita tool lifecycle;
- получение canvas/image coordinates;
- cursor и selection modifier behavior;
- сбор lasso/freehand region input на canvas;
- отрисовку lasso/preparation/result preview в `paint()`;
- применение mask как обычного выделения Krita;
- инвалидацию interaction state при смене документа, tool или изображения.

Python часть отвечает за:

- loader wrapper для native library;
- worker process entrypoint;
- model registry;
- download/cache orchestration;
- PyTorch/SAM2 inference в worker process.

Граница между C++ tool и worker должна быть небольшой: C++ передаёт snapshot
изображения и lasso/region prompt на подготовку, worker возвращает result mask,
опциональную `segment_map`, masks и metadata, после чего C++ выполняет preview и
selection apply.

## Canvas Integration

Для MVP canvas/tool слой должен быть C++/hybrid, а не Python-only overlay.

Overlay/preview не является отдельным floating window или QWidget. Это
состояние отрисовки нативного Krita tool-а:

- по умолчанию overlay ничего не рисует;
- во время lasso gesture tool рисует нативный контур области;
- пока worker готовит результат, tool показывает preparing state;
- после результата tool рисует полупрозрачную mask preview в `paint()`;
- repaint ограничивается bbox preview/region там, где это практически возможно.

Full-image prepared hover/click spike может использовать label/segment map
формат `pixel -> segment_id` для O(1) hit-test. Для основного MVP lasso workflow
authoritative result может быть single mask с bbox/metadata; full-canvas
`segment_map` остаётся допустимым транспортным форматом, если это упрощает C++
selection apply.

## Selection Behavior

Поведение lasso-выделения должно соответствовать обычным инструментам выделения
Krita:

- Завершение lasso gesture без modifier: replace текущего выделения.
- Shift + завершение lasso: add.
- Alt/Ctrl + завершение lasso: subtract/intersect/toggle в соответствии с нативными
  conventions Krita и доступными `SelectionAction`.

Точное сопоставление Alt/Ctrl нужно брать из Krita selection tool behavior, а не
изобретать отдельно.

## Preparing UI States

Пока worker занят, UI не должен выглядеть зависшим:

- На canvas или в tool options нужно показывать `Preparing smart selection...`.
- Нужно показывать текущий stage: `checking_runtime`, `creating_runtime`,
  `installing_runtime_dependencies`, `validating_runtime`, `resolving_model`,
  `checking_model_cache`, `downloading_model`, `verifying_model`,
  `reading_image`, `preprocessing_image`, `loading_model`, `running_sam`,
  `building_selection_mask`.
- Progress должен появляться быстро, даже если сначала он indeterminate.
- Должна быть кнопка Cancel.
- Если пользователь сменил tool, документ или закрыл изображение, job должен
  отменяться или уходить в безопасный background state без применения результата
  к неправильному документу.
- Ошибка подготовки должна показывать короткое понятное сообщение и действия из
  support flow.

## Product Direction

Первоначальный canvas spike доказал full-image prepared hover/click overlay, но
качество automatic masks на сложных краях оказалось главным ограничением.
Основной MVP workflow переориентирован на lasso/region-prompted segmentation:
пользователь задаёт область интереса, SAM2 работает на crop/region prompt, а
результат применяется как выделение без дополнительного click-to-select шага.

## Packaging Target

MVP таргетирует конкретную версию Krita, а не широкий диапазон версий.

Начальный target:

- Linux.
- Krita 5.3.1.
- Native C++ module/library.
- Python extension как loader wrapper.

Поддержка нескольких версий Krita и кросс-версионная binary compatibility не
являются целью первого MVP.

## Current Implementation Status

Уже реализовано:

- native Smart Select tool registration;
- lasso/freehand Smart Select input;
- lasso/region-prompted SAM2 selection apply;
- Python loader wrapper;
- worker-backed lasso selection через `QProcess`;
- flattened snapshot export as RGB PNG plus alpha mask;
- downscaled `128x128` snapshot fingerprint;
- Tool Options status/progress/cancel block;
- ModelRegistry для `sam2.1_hiera_large`;
- worker-side checkpoint download/cache/verify;
- worker-managed system Python venv/cache для SAM2 runtime;
- real SAM2.1 Large prompted selection;
- conversion from SAM masks to full-canvas `uint8` selection mask;
- runtime diagnostics foundation;
- type checking and code tests.

Ещё не реализовано:

- model picker and lighter model presets;
- candidate mask debug export;
- persistent logs and support dialog.
