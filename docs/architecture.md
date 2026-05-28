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
- хранение текущего `hoverSegmentId`;
- O(1) hit-test по prepared segment map;
- отрисовку hover-подсветки в `paint()`;
- применение mask как обычного выделения Krita;
- инвалидацию interaction state при смене документа, tool или изображения.

Python часть отвечает за:

- loader wrapper для native library;
- worker process entrypoint;
- model registry;
- download/cache orchestration;
- PyTorch/SAM2 inference в worker process.

Граница между C++ tool и worker должна быть небольшой: C++ передаёт snapshot
изображения на подготовку, worker возвращает `segment_map`, masks и metadata,
после чего C++ выполняет hit-test, overlay и selection apply.

## Canvas Integration

Для MVP canvas/tool слой должен быть C++/hybrid, а не Python-only overlay.

Overlay не является отдельным floating window или QWidget. Это состояние
отрисовки нативного Krita tool-а:

- по умолчанию overlay ничего не рисует;
- при движении курсора tool переводит позицию в image coordinates;
- по координате курсора tool читает `segment_id` из segment map;
- если `segment_id` изменился, tool обновляет `hoverSegmentId`;
- tool запрашивает repaint старой и новой области сегмента;
- в `paint()` tool рисует полупрозрачную красную mask только для текущего
  hover-сегмента.

Для MVP предпочтителен label/segment map формат `pixel -> segment_id`, чтобы
hover hit-test был O(1). Перебор всех masks на каждое движение мыши допустим
только как временный прототип для маленьких изображений.

## Selection Behavior

Поведение должно соответствовать обычным инструментам выделения Krita:

- Click: replace текущего выделения.
- Shift + click: add.
- Alt/Ctrl + click: subtract/intersect/toggle в соответствии с нативными
  conventions Krita и доступными `SelectionAction`.

Точное сопоставление Alt/Ctrl нужно брать из Krita selection tool behavior, а не
изобретать отдельно.

## Preparing UI States

Пока worker занят, UI не должен выглядеть зависшим:

- На canvas или в tool options нужно показывать `Preparing smart segments...`.
- Нужно показывать текущий stage: `checking_runtime`, `creating_runtime`,
  `installing_runtime_dependencies`, `validating_runtime`, `resolving_model`,
  `checking_model_cache`, `downloading_model`, `verifying_model`,
  `reading_image`, `preprocessing_image`, `loading_model`, `running_sam`,
  `building_segment_map`, `building_hover_index`.
- Progress должен появляться быстро, даже если сначала он indeterminate.
- Должна быть кнопка Cancel.
- Если пользователь сменил tool, документ или закрыл изображение, job должен
  отменяться или уходить в безопасный background state без применения результата
  к неправильному документу.
- Ошибка подготовки должна показывать короткое понятное сообщение и действия из
  support flow.

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
- canvas hover overlay для prepared segment map;
- click-to-selection;
- Python loader wrapper;
- worker-backed mock segmentation через `QProcess`;
- flattened snapshot export as RGB PNG plus alpha mask;
- downscaled `128x128` snapshot fingerprint;
- exact in-memory cache reuse by size/model/algorithm/fingerprint;
- debounced rebuild on active image updates;
- Tool Options status/progress/cancel block;
- ModelRegistry для `sam2.1_hiera_large`;
- worker-side checkpoint download/cache/verify;
- worker-managed system Python venv/cache для SAM2 runtime;
- real SAM2.1 Large automatic mask generation;
- conversion from SAM masks to full-canvas `uint16` segment map;
- runtime diagnostics foundation;
- type checking and code tests.

Ещё не реализовано:

- model picker and lighter model presets;
- persistent logs and support dialog.
