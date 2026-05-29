# Worker Protocol

## Transport

MVP использует `QProcess` + stdio NDJSON как control channel.

- stdin worker-а принимает one-line JSON requests.
- stdout worker-а отдаёт one-line JSON progress/results/errors.
- stderr worker-а считается логом worker process.
- Большие данные передаются через временные файлы, не через JSON.

## Large Data Files

C++ writes:

- flattened RGB snapshot PNG;
- document alpha mask `.npy`, dtype `|u1`;
- current selection region mask `.npy`, dtype `|u1`.

Worker writes:

- `selection_mask.npy`, dtype `|u1`, full canvas size.

After successful read, C++ may delete temporary files.

## Requests

### `prepare_selection`

The only inference request in the MVP path:

```json
{"id":1,"method":"prepare_selection","params":{"image_path":"/tmp/input.png","alpha_mask_path":"/tmp/alpha_mask.npy","region_mask_path":"/tmp/region_mask.npy","region_bbox":[100,120,420,360],"model_id":"birefnet_hr_matting","algorithm_version":"birefnet-hr-matting-lasso-v1","fingerprint":"...","canvas_width":1024,"canvas_height":768,"model_cache_dir":"/home/user/.local/share/krita/smart_select/models","runtime_cache_dir":"/home/user/.local/share/krita/smart_select/runtime"}}
```

Parameters:

- `image_path`: RGB snapshot input.
- `alpha_mask_path`: document alpha mask. `0` means not selectable; `1..255`
  means visible/selectable.
- `region_mask_path`: current selection region mask. `0` means outside the
  selected region; `1..255` means selected and may preserve soft selection
  values from native Krita selection editing.
- `region_bbox`: `[x, y, width, height]` in canvas pixel coordinates.
- `model_id`: currently `birefnet_hr_matting`.
- `algorithm_version`: currently `birefnet-hr-matting-lasso-v1`.
- `fingerprint`: snapshot/region fingerprint for stale-result context.
- `canvas_width`, `canvas_height`: full canvas size in pixels.
- `model_cache_dir`: writable model snapshot cache.
- `runtime_cache_dir`: writable managed runtime cache.

Removed protocol paths:

- `prepare_segments`;
- `mock` model id;
- `refiner_id`;
- `refiner_cache_dir`;
- `segment_map_path`;
- `segments_path`.

### `diagnostics`

```json
{"id":2,"method":"diagnostics","params":{"model_cache_dir":"/home/user/.local/share/krita/smart_select/models","runtime_cache_dir":"/home/user/.local/share/krita/smart_select/runtime"}}
```

Diagnostics do not run inference and must not inspect image data.

### `cancel`

```json
{"id":2,"method":"cancel","params":{"job_id":1}}
```

Cancellation is responsive during runtime/model setup and stale-safe during GPU
inference.

## Progress

```json
{"id":1,"event":"progress","stage":"running_alpha_model","progress":null}
```

Stable stages:

- `checking_runtime`
- `creating_runtime`
- `installing_runtime_dependencies`
- `validating_runtime`
- `resolving_model`
- `checking_model_cache`
- `downloading_model`
- `verifying_model`
- `loading_model`
- `running_alpha_model`
- `building_alpha_mask`

Unknown extra fields are ignored by C++.

## Success

```json
{"id":1,"result":{"selection_mask_path":"/tmp/selection_mask.npy","metadata":[{"id":1,"bbox":[0,0,100,100],"area":10000,"score":1.0}],"model":{"id":"birefnet_hr_matting","title":"BiRefNet HR Matting","cache_state":"verified","checkpoint_bytes":444473596},"runtime":{"version":"birefnet-runtime-v1","cache_state":"ready","device":"cuda","torch_version":"2.6.0+cu124","cuda_available":true}}}
```

C++ must trust only `selection_mask.npy` for selection apply. Metadata is useful
for logs/UI but is not authoritative mask data.

## Selection Mask Contract

- Format: NumPy `.npy`.
- Dtype: `|u1`.
- Shape: `[canvas_height, canvas_width]`.
- Value `0`: not selected.
- Values `1..254`: soft/partial selection alpha.
- Value `255`: fully selected.
- Coordinate convention: image pixel coordinates, origin top-left.
- C++ must preserve raw `0..255` values.

The worker clips model alpha to:

```text
final = birefnet_alpha & selection_region & document_alpha
```

There is no hidden grow/feather/erosion step in the worker.

## Errors

```json
{"id":1,"error":{"type":"InferenceFailed","message":"alpha extraction failed"}}
```

Expected user-safe error types:

- `RuntimeCacheDirMissing`
- `RuntimeBootstrapError`
- `RuntimeDependencyError`
- `RuntimeValidationError`
- `ModelCacheDirMissing`
- `ModelDownloadError`
- `ModelVerificationError`
- `UnknownModel`
- `InferenceOutOfMemory`
- `InferenceFailed`
- `EmptyRegion`
- `EmptySelection`
- `WorkerBusy`
- `ProtocolError`
