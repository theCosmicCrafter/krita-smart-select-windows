# Observability

## Goals

Плагин должен быть максимально прозрачен при ошибках и долгих операциях.

Цель observability:

- сделать поддержку возможной без угадывания по скриншотам;
- не просить пользователя вручную искать системные логи;
- иметь короткий безопасный Diagnostic Report;
- иметь полный лог для сложных случаев;
- не утекать содержимым изображений, pixel data, masks или секретами.

## Required Log Coverage

Логи должны покрывать:

- версию плагина;
- версию Krita;
- версию Python в Krita;
- путь к venv/worker environment;
- ОС и архитектуру;
- доступные устройства inference: CPU, CUDA и другие backend, если появятся;
- выбранную модель и её registry id;
- тип job-а: full-image spike или lasso/region selection;
- lasso/region bbox и размеры mask, но не пиксельные данные mask;
- состояние model cache;
- этапы загрузки и проверки модели;
- start/progress/cancellation/result inference jobs;
- ошибки worker-процесса;
- stderr/stdout worker-процесса;
- ошибки Krita API при создании selection/layer.

## Structured Events

Нужно использовать структурированные записи событий, чтобы их можно было
фильтровать:

```text
[SmartSelect] event.name=selection.start event.domain=smart_select severity=info component=krita_plugin job.id=12 job.type=lasso_selection model.id=sam2.1_hiera_large
[SmartSelectWorker] event.name=sam.predict.result event.domain=smart_select severity=info component=sam2 job.id=12 model.id=sam2.1_hiera_large device.type=cuda duration_ms=531 masks.count=3
```

Stable event names:

- `setup.tool_factory.register`
- `setup.tool_action.inject`
- `setup.native_runtime`
- `tool.activate`
- `tool.deactivate`
- `tool.destroy`
- `tool.image.updated`
- `runtime.cache.path`
- `runtime.cache.failed`
- `runtime.setup.start`
- `runtime.setup.progress`
- `runtime.setup.failed`
- `runtime.setup.result`
- `model.download.start`
- `model.download.failed`
- `model.download.progress`
- `model.download.retry`
- `model.download.result`
- `model.cache.path`
- `model.cache.failed`
- `worker.environment.sanitized`
- `worker.start`
- `worker.start.failed`
- `worker.stderr`
- `worker.crashed`
- `lasso.finish`
- `lasso.region_mask.created`
- `lasso.region_mask.failed`
- `snapshot.capture.start`
- `snapshot.capture.failed`
- `snapshot.files.failed`
- `selection.start`
- `selection.progress`
- `selection.cancelled`
- `selection.failed`
- `selection.result`
- `selection.result.ignored_stale`
- `selection.error.ignored_stale`
- `selection.apply.failed`
- `sam.generate.start`
- `sam.generate.result`
- `sam.predict.start`
- `sam.predict.result`
- `selection_mask.prior.result`
- `segment_map.result`
- `selection_mask.result`
- `selection_mask.fallback`
- `trimap.result`
- `refiner.download.start`
- `refiner.download.result`
- `refiner.download.failed`
- `mam.load.start`
- `mam.load.result`
- `mam.refine.result`
- `refinement.gate.result`
- `refinement.rejected`
- `refinement.failed`
- `selection.debug_export.metrics`
- `selection.debug_export.result`
- `selection.debug_export.failed`
- `region_input.result`
- `temp_file.cleaned`
- `temp_file.cleanup_failed`
- `temp_file.kept_for_debug`
- `temp_dir.cleaned`

Stable event names важнее красивого текста. User-facing message может быть
отдельным полем.

## OpenTelemetry Compatibility

MVP logs are OpenTelemetry-compatible by schema, but the plugin does not use an
OpenTelemetry SDK, OTLP exporter or network telemetry.

Required common fields:

- `event.name`
- `event.domain=smart_select`
- `severity`
- `component`

Preferred dotted fields:

- `job.id`
- `job.type`
- `job.stage`
- `model.id`
- `model.cache.state`
- `model.checkpoint.bytes`
- `runtime.version`
- `runtime.cache.state`
- `device.type`
- `cuda.available`
- `duration_ms`
- `masks.count`
- `region.bbox`
- `region.area`
- `selection.area`
- `selection.bbox`
- `error.type`
- `error.message`
- `error.reason`

The current transport remains human-readable key-value text through `qInfo`,
`qWarning` and worker `stderr`. Persistent JSON Lines logs and OTLP export are
future work. OTLP export must be explicit opt-in and must not be added without
a product decision.

New logs must use `event.name=...`, not `event=...`.

Worker-side failure events must include enough non-sensitive detail to debug
the root cause without reproducing immediately. In particular,
`model.download.failed` must include:

- `model.id`;
- stable `error.type`;
- short `error.reason` / exception representation.

`model.download.retry` should include the same failure shape plus the retry
attempt and requested byte range. This keeps transient network failures visible
without turning them into user-facing errors.

The `reason` must not include tokens, environment variables, image paths,
document paths, pixel data or masks.

## Diagnostic Report

Diagnostic Report нужен не вместо полноценных логов, а как безопасный и быстрый
первый слой диагностики. Он должен быть достаточно коротким, чтобы его можно
было автоматически вставить в issue body или скопировать без риска утечки
лишних данных.

Краткий Diagnostic Report должен включать:

- версию плагина;
- версию Krita;
- ОС и архитектуру;
- версию Python в Krita;
- тип установки Krita, если его можно определить: AppImage, system package,
  Flatpak, Windows/macOS bundle;
- выбранную модель и её registry id;
- device/backend: CPU, CUDA, CUDA version, GPU name, если доступно;
- состояние модели: not downloaded, downloading, verified, load failed, loaded;
- последнее действие пользователя;
- последний failed event id;
- последний exception type и короткий message;
- путь к локальному log file, с редактированием домашней директории;
- несколько последних структурированных событий без stack trace.

## Full Log

Полный лог остаётся отдельным артефактом и нужен для сложных случаев:

- полный stack trace;
- подробный worker lifecycle;
- подробный download/inference progress;
- stderr/stdout worker-процесса;
- детали model loading;
- ошибки Krita API.

## Debug Segment Maps And Selection Masks

По умолчанию временные snapshot/result файлы удаляются после успешного чтения
C++ tool-ом. Для локального дебага можно запустить Krita с:

```bash
SMART_SELECT_KEEP_DEBUG_FILES=1 /path/to/krita.AppImage
```

В этом режиме логи `temp_file.kept_for_debug`, `selection.debug_export.result`
и `inference.result` покажут пути
к `snapshot.png`, `alpha_mask.npy`, `region_mask.npy`, `selection_mask.npy`,
`segment_map.npy` и `segments.npz`, если эти файлы участвуют в текущем job-е.
Worker также сохраняет debug-копии `sam_prior_mask.png`, `trimap.png`,
`trimap_core_mask.png`, `trimap_unknown_mask.png`, `trimap_support_mask.png`,
`mam_alpha.png`, `final_alpha.png`, `mam_added_vs_prior.png` и
`mam_removed_vs_prior.png` для сравнения coarse prior, trimap guidance и
refined alpha.

`selection.debug_export.metrics` дополнительно пишет численные диагностические
поля:

- `selection.prior.area`
- `selection.region.area`
- `selection.alpha_sum`
- `selection.soft_pixels`
- `refinement.added_vs_prior_pixels`
- `refinement.removed_vs_prior_pixels`
- `refinement.outside_region_pixels`
- `refinement.mam_full_pixels`
- `refinement.mam_soft_pixels`
- `refinement.rejected`
- `trimap.core.area`
- `trimap.support.area`
- `trimap.unknown.area`

Эти счетчики разрешены в логах, потому что они не содержат image contents,
pixel arrays или mask bytes.

`segment_map.npy` можно отрендерить в PPM:

```bash
./scripts/render-segment-map.py /tmp/krita-smart-select-.../segment_map.npy /tmp/segments.ppm
```

Для отдельного segment id:

```bash
./scripts/render-segment-map.py /tmp/krita-smart-select-.../segment_map.npy /tmp/segment-12.ppm --segment-id 12
```

Debug files can contain document-derived image data and must not be attached to
public issues unless the user explicitly wants to share the image.

## Redaction Rules

Логи не должны включать:

- содержимое изображений;
- pixel data;
- masks;
- tokens;
- секретные переменные окружения.

Перед копированием отчёта в issue нужно по возможности редактировать:

- домашнюю директорию пользователя;
- абсолютные пути к документам;
- значения переменных окружения с секретами.

## Support UI Actions

В UI настроек или support dialog должен быть раздел диагностики с кнопками:

- Copy Diagnostic Report
- Copy Full Log
- Open Logs Folder
- Open GitHub Issue

## GitHub Issue Flow

Кнопка `Open GitHub Issue` должна показывать pop-up с двумя вариантами:

- Manual
- Auto (requires GitHub CLI)

REST API GitHub для создания issue не рассматривается. Плагин не должен сам
получать, хранить или обновлять GitHub token.

### Manual Flow

- Плагин открывает браузер с prefilled issue template.
- В шаблон можно передавать краткий отчёт через query params.
- Полный лог лучше предлагать скопировать отдельно, чтобы не упереться в лимиты
  URL и случайно не отправить лишние данные.
- Пользователь сам отправляет issue в браузере.

### Auto Flow Через GitHub CLI

- Плагин проверяет наличие `gh` в системе.
- Плагин проверяет авторизацию через `gh auth status`.
- Если `gh` не найден, плагин показывает ошибку: нужно установить GitHub CLI.
- Если `gh` найден, но не авторизован, плагин показывает ошибку: нужно
  выполнить `gh auth login` и повторить действие.
- Если `gh` найден и авторизован, плагин показывает preview issue body.
- После явного подтверждения пользователя плагин создаёт issue через
  `gh issue create --repo OWNER/REPO --title ... --body-file ...`.

Auto flow должен оставаться optional.

## Current Implementation Status

Текущая реализация уже пишет structured events через `qInfo`, `qWarning` и
Python `print`.

Ещё не реализовано:

- persistent plugin log file;
- Diagnostic Report UI;
- Copy Full Log;
- Open Logs Folder;
- Open GitHub Issue.
