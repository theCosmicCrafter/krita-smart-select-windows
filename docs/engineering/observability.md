# Observability

## Goals

Плагин должен быть прозрачен при ошибках и долгих операциях, не раскрывая
данные документа.

Observability должна помогать:

- понять, где job остановился;
- отличить runtime/model setup от inference failure;
- диагностировать CUDA/CPU поведение;
- не логировать image contents, pixel data, mask bytes, tokens or secrets.

## OpenTelemetry Compatibility

Logs are OpenTelemetry-compatible by schema, but the plugin does not use an
OpenTelemetry SDK, OTLP exporter or network telemetry.

New logs must use:

```text
event.name=...
```

not:

```text
event=...
```

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
- `model.repo_id`
- `model.revision`
- `model.cache.state`
- `model.checkpoint.bytes`
- `runtime.version`
- `runtime.cache.state`
- `device.type`
- `cuda.available`
- `duration_ms`
- `image.width`
- `image.height`
- `region.bbox`
- `region.area`
- `selection.area`
- `selection.alpha_sum`
- `selection.soft_pixels`
- `error.type`
- `error.message`
- `error.reason`

The current transport remains human-readable key-value text through `qInfo`,
`qWarning` and worker `stderr`. Persistent JSON Lines logs or OTLP export must
be explicit opt-in product decisions.

## Stable Events

C++ events:

- `setup.tool_factory.register`
- `setup.tool_action.inject`
- `setup.native_runtime`
- `tool.activate`
- `tool.deactivate`
- `tool.destroy`
- `tool.image.updated`
- `runtime.cache.path`
- `runtime.cache.failed`
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
- `temp_file.cleaned`
- `temp_file.cleanup_failed`
- `temp_file.kept_for_debug`
- `temp_dir.cleaned`

Worker events:

- `runtime.setup.result`
- `alpha_model.cache.start`
- `alpha_model.cache.result`
- `alpha_model.cache.failed`
- `alpha_model.load.start`
- `alpha_model.load.result`
- `alpha_model.predict.start`
- `alpha_model.predict.result`
- `alpha_model.predict.failed`
- `alpha_model.memory.released`
- `alpha_mask.result`
- `selection.debug_export.result`
- `selection.debug_export.failed`

## Debug Files

By default temporary snapshot/result files are cleaned after successful read.
For local debugging:

```bash
./scripts/clean-debug-data.sh
SMART_SELECT_KEEP_DEBUG_FILES=1 /path/to/krita.AppImage
```

Debug export files:

- `crop_rgb.png`
- `crop_region_mask.png`
- `crop_alpha_mask.png`
- `birefnet_alpha_raw.png`
- `final_alpha.png`
- `metadata.json`

Debug files can contain document-derived image data and must not be attached to
public issues unless the user explicitly wants to share the image.

## Redaction Rules

Logs and future Diagnostic Reports must not include:

- image contents;
- pixel arrays;
- mask bytes;
- environment variables;
- tokens or secrets.

User-facing diagnostic output should redact home/document paths where possible.
Developer logs may include temporary paths while debugging.
