# Worker Protocol

## Transport

MVP использует `QProcess` + stdio NDJSON как control channel.

- C++ запускает worker через `QProcess`.
- stdin worker-а принимает one-line JSON requests.
- stdout worker-а отдаёт one-line JSON events/results/errors.
- stderr worker-а считается логом worker process.
- Большие данные не передаются через JSON.

Большие массивы передаются через временные файлы:

- C++ сохраняет flattened snapshot как RGB PNG и передаёт worker путь.
- C++ сохраняет alpha/background mask как `uint8` `.npy` и передаёт worker
  путь.
- Python worker пишет result mask/`segment_map` и metadata во временные
  `.npy`/`.npz` файлы.
- Worker возвращает пути вместе с metadata.
- C++ tool читает результат в память.
- После успешного чтения временные файлы можно удалить.

Shared memory можно рассмотреть позже, если временные файлы окажутся узким
местом. Для MVP она не нужна.

## Requests

### `prepare_selection`

Primary MVP request for the lasso workflow:

```json
{"id":1,"method":"prepare_selection","params":{"image_path":"/tmp/input.png","alpha_mask_path":"/tmp/alpha_mask.npy","region_mask_path":"/tmp/region_mask.npy","region_bbox":[100,120,420,360],"model_id":"sam2.1_hiera_large","refiner_id":"mam_sam_vith","algorithm_version":"sam2.1-lasso-trimap-mam-v1","fingerprint":"...","canvas_width":1024,"canvas_height":768,"model_cache_dir":"/home/user/.local/share/krita/smart_select/models","runtime_cache_dir":"/home/user/.local/share/krita/smart_select/runtime","refiner_cache_dir":"/home/user/.local/share/krita/smart_select/refiners"}}
```

Parameters:

- `image_path`: путь к RGB snapshot input.
- `alpha_mask_path`: путь к `uint8` `.npy` alpha/background mask. Значение `0`
  означает fully transparent/background, значения `1..255` считаются видимыми.
- `region_mask_path`: путь к `uint8` `.npy` mask нарисованной lasso-области.
  Значение `0` означает outside region, значения `1..255` считаются region
  prompt.
- `region_bbox`: `[x, y, width, height]` bounds lasso-области в canvas
  coordinates. Worker может использовать bbox + padding для crop inference.
- `model_id`: registry id модели, например `sam2.1_hiera_large`.
- `refiner_id`: optional refiner id. Current default for real model requests is
  `mam_sam_vith`.
- `algorithm_version`: версия алгоритма подготовки selection mask.
- `fingerprint`: fingerprint snapshot-а и region input.
- `canvas_width`: ширина canvas в image pixels.
- `canvas_height`: высота canvas в image pixels.
- `model_cache_dir`: writable directory для checkpoint cache.
- `runtime_cache_dir`: writable directory для managed Python venv/cache.
- `refiner_cache_dir`: writable directory для MAM/SAM1 refiner checkpoints.
  Обязателен для `mam_sam_vith`.

`prepare_selection` возвращает mask, применяемую как Krita selection. Full-image
hover hit-test не является частью этого request-а.

### `prepare_segments`

Legacy/spike request for full-image prepared hover mode:

```json
{"id":1,"method":"prepare_segments","params":{"image_path":"/tmp/input.png","alpha_mask_path":"/tmp/alpha_mask.npy","model_id":"sam2.1_hiera_large","algorithm_version":"sam2.1-auto-mask-quality-v1","fingerprint":"...","canvas_width":1024,"canvas_height":768,"model_cache_dir":"/home/user/.local/share/krita/smart_select/models","runtime_cache_dir":"/home/user/.local/share/krita/smart_select/runtime"}}
```

Parameters:

- `image_path`: путь к RGB snapshot input.
- `alpha_mask_path`: путь к `uint8` `.npy` alpha/background mask. Значение `0`
  означает fully transparent/background, значения `1..255` считаются видимыми.
- `model_id`: registry id модели, например `sam2.1_hiera_large` или `mock`.
- `algorithm_version`: версия алгоритма подготовки segment map.
- `fingerprint`: fingerprint snapshot-а. Текущий MVP использует downscaled
  `128x128` hash.
- `canvas_width`: ширина canvas в image pixels.
- `canvas_height`: высота canvas в image pixels.
- `model_cache_dir`: writable directory для checkpoint cache. Обязателен для
  real model ids и может отсутствовать для `mock`.
- `runtime_cache_dir`: writable directory для managed Python venv/cache.
  Обязателен для real model ids и может отсутствовать для `mock`.

Текущий MVP real-inference path использует `prepare_selection`,
`model_id="sam2.1_hiera_large"` и
`algorithm_version="sam2.1-lasso-trimap-mam-v1"`. Legacy full-image
`prepare_segments` path сохраняет собственную algorithm version только для
spike/dev compatibility.

### `diagnostics`

```json
{"id":2,"method":"diagnostics","params":{"model_cache_dir":"/home/user/.local/share/krita/smart_select/models","runtime_cache_dir":"/home/user/.local/share/krita/smart_select/runtime"}}
```

Возвращает Python/platform/torch/model-cache сведения для будущего Diagnostic
Report. Этот request не запускает inference и не читает image input.

### `cancel`

```json
{"id":2,"method":"cancel","params":{"job_id":1}}
```

Parameters:

- `job_id`: id request-а, который нужно отменить или безопасно игнорировать.

## Events

### Progress

```json
{"id":1,"event":"progress","stage":"generating_segments","progress":0.42}
```

Fields:

- `id`: request id.
- `event`: `progress`.
- `stage`: stable stage name.
- `progress`: number from `0.0` to `1.0`, or `null` for indeterminate progress.

Expected shared stages:

- `checking_runtime`
- `creating_runtime`
- `installing_runtime_dependencies`
- `validating_runtime`
- `resolving_model`
- `checking_model_cache`
- `downloading_model`
- `verifying_model`
- `reading_image`
- `preprocessing_image`
- `loading_model`
- `running_sam`
- `building_selection_mask`

Legacy full-image stages:

- `building_segment_map`
- `building_hover_index`

`downloading_model` может добавлять extra fields:

```json
{"id":1,"event":"progress","stage":"downloading_model","progress":0.42,"bytes_done":377194116,"bytes_total":898089053}
```

C++ может игнорировать unknown fields.

MAM/refiner stages:

- `resolving_refiner`
- `checking_refiner_cache`
- `downloading_refiner`
- `verifying_refiner`
- `loading_refiner`
- `running_mam`
- `building_alpha_mask`

## Success

```json
{"id":1,"result":{"selection_mask_path":"/tmp/selection_mask.npy","segment_map_path":"/tmp/segment_map.npy","segments_path":"/tmp/segments.npz","metadata":[{"id":1,"bbox":[0,0,100,100],"area":10000,"score":1.0}],"model":{"id":"sam2.1_hiera_large","title":"SAM2.1 Hiera Large","cache_state":"verified","checkpoint_bytes":898089053},"runtime":{"version":"sam2-runtime-v3","cache_state":"ready","device":"cuda","torch_version":"2.6.0+cu124","cuda_available":true,"sam2_source_ref":"2b90b9f5ceec907a1c18123530e92e794ad901a4"},"refiner":{"id":"mam_sam_vith","title":"MAM SAM ViT-H","cache_state":"verified","device":"cuda","sam_checkpoint_bytes":2564550879,"mam_checkpoint_bytes":32785332}}}
```

Fields:

- `selection_mask_path`: optional path to `.npy` single selection mask for
  lasso workflow.
- `segment_map_path`: optional path to `.npy` label map.
- `segments_path`: path to `.npz` segment data.
- `metadata`: segment metadata list.
- `model`: optional model readiness metadata for logs/diagnostics.
- `runtime`: optional runtime/inference metadata for logs/diagnostics.
- `refiner`: optional refiner metadata for logs/diagnostics.

Current lasso MVP rule: C++ authoritative source for selection apply is
`selection_mask.npy` when present. Full-image hover/click spike can still use
`segment_map.npy` as authoritative source for hit-testing. Worker metadata is
useful for logs/UI but must not be trusted as mask data.

For `prepare_selection`, worker-side mask choice is intentionally not a raw
“highest SAM score wins” decision. The worker builds a recall-oriented SAM prior:

- run the predictor with the lasso bbox and generated points;
- when negative points exist, also run positive-only and box-only prompt
  variants with lower score weight;
- score candidates against the lasso region and generated prompt points;
- penalize candidates that mostly hug the lasso boundary or hit negative points;
- merge compatible candidates so missing object parts can be recovered;
- fill small enclosed holes;
- hard-clip the result to lasso and alpha.

The current primary path refines that prior with MAM SAM ViT-H and writes a
soft alpha selection mask. Before MAM runs, the worker converts the SAM prior
into a trimap:

- `core`: eroded SAM prior, always preserved as foreground;
- `unknown`: narrow support band around the prior where MAM may refine alpha;
- `background`: everything outside support, lasso or document alpha.

MAM output is accepted only if it stays close to the SAM prior by area and by
added/removed pixels. If the gate rejects the matte, the worker falls back to
the SAM prior and marks the refiner payload as `rejected_mam`. If MAM inference
fails after a valid prior exists, the worker may fall back to the SAM prior and
mark the payload as `fallback_sam_prior`. Invalid or unavailable H checkpoint
metadata is a setup error and must not silently fall back to ViT-B.

## Error

```json
{"id":1,"error":{"type":"ModelLoadError","message":"short user-safe message","details_event_id":"..."}}
```

Fields:

- `type`: stable machine-readable error type.
- `message`: short user-safe message.
- `details_event_id`: optional id for matching the error to full logs.

Expected model readiness error types:

- `UnknownModel`
- `RuntimeCacheDirMissing`
- `RuntimeBootstrapError`
- `RuntimeDependencyError`
- `RuntimeValidationError`
- `ModelCacheDirMissing`
- `ModelDownloadError`
- `ModelVerificationError`
- `ModelDownloadCancelled`
- `InferenceDeviceError`
- `InferenceOutOfMemory`
- `InferenceFailed`
- `ImageTooLarge`
- `WorkerBusy`
- `RefinerCacheDirMissing`
- `RefinerDownloadError`
- `RefinerVerificationError`
- `RefinerRuntimeValidationError`
- `RefinerUnavailable`
- `MattingFailed`
- `MattingOutOfMemory`

## Segment Map Contract

- Format: NumPy `.npy`.
- Current dtype: `<u2` / `uint16`.
- Shape: `[canvas_height, canvas_width]`.
- Value `0`: no segment/background.
- Value `>0`: segment id.
- Coordinate convention: image pixel coordinates, origin top-left.

If segment count can exceed `65535`, the contract must be upgraded to `uint32`
explicitly on both C++ and worker sides.

## Alpha Mask Contract

- Format: NumPy `.npy`.
- Current dtype: `|u1` / `uint8`.
- Shape: `[canvas_height, canvas_width]`.
- Value `0`: fully transparent/background, must not be selectable.
- Value `1..255`: visible/semi-visible, may be selectable.
- Coordinate convention: image pixel coordinates, origin top-left.

## Region Mask Contract

- Format: NumPy `.npy`.
- Current dtype: `|u1` / `uint8`.
- Shape: `[canvas_height, canvas_width]`.
- Value `0`: outside lasso/region prompt.
- Value `1..255`: inside lasso/region prompt.
- Coordinate convention: image pixel coordinates, origin top-left.

## Selection Mask Contract

- Format: NumPy `.npy`.
- Current dtype: `|u1` / `uint8`.
- Shape: `[canvas_height, canvas_width]`.
- Value `0`: not selected.
- Value `1..254`: soft/partial selection alpha.
- Value `255`: fully selected.
- Coordinate convention: image pixel coordinates, origin top-left.

## Segment Metadata Contract

```json
{
  "id": 1,
  "bbox": [x, y, width, height],
  "area": 1234,
  "score": 0.97
}
```

Required fields:

- `id`
- `bbox`
- `area`

Optional fields:

- `score`

## Current Implementation Status

Current worker implementation resolves and verifies `sam2.1_hiera_large` and
uses `prepare_selection` for lasso/region-prompted selection as the primary GUI
path. The full-image `prepare_segments` path remains available for spike/dev
compatibility.
