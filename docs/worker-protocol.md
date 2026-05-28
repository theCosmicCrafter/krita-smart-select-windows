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
- Python worker пишет `segment_map` и masks во временные `.npy`/`.npz` файлы.
- Worker возвращает пути вместе с metadata.
- C++ tool читает результат в память.
- После успешного чтения временные файлы можно удалить.

Shared memory можно рассмотреть позже, если временные файлы окажутся узким
местом. Для MVP она не нужна.

## Requests

### `prepare_segments`

```json
{"id":1,"method":"prepare_segments","params":{"image_path":"/tmp/input.png","alpha_mask_path":"/tmp/alpha_mask.npy","model_id":"sam2_hiera_tiny","algorithm_version":"sam2-auto-mask-v1","fingerprint":"...","canvas_width":1024,"canvas_height":768}}
```

Parameters:

- `image_path`: путь к RGB snapshot input.
- `alpha_mask_path`: путь к `uint8` `.npy` alpha/background mask. Значение `0`
  означает fully transparent/background, значения `1..255` считаются видимыми.
- `model_id`: registry id модели, например `sam2_hiera_tiny` или `mock`.
- `algorithm_version`: версия алгоритма подготовки segment map.
- `fingerprint`: fingerprint snapshot-а. Текущий MVP использует downscaled
  `128x128` hash.
- `canvas_width`: ширина canvas в image pixels.
- `canvas_height`: высота canvas в image pixels.

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

Expected MVP stages:

- `loading_model`
- `reading_image`
- `generating_segments`
- `building_hover_index`

## Success

```json
{"id":1,"result":{"segment_map_path":"/tmp/segment_map.npy","segments_path":"/tmp/segments.npz","metadata":[{"id":1,"bbox":[0,0,100,100],"area":10000,"score":1.0}]}}
```

Fields:

- `segment_map_path`: path to `.npy` label map.
- `segments_path`: path to `.npz` segment data.
- `metadata`: segment metadata list.

Current MVP rule: C++ authoritative source for hover/click is
`segment_map.npy`. Worker metadata is useful for logs/UI but must not be
trusted for hit-testing.

## Error

```json
{"id":1,"error":{"type":"ModelLoadError","message":"short user-safe message","details_event_id":"..."}}
```

Fields:

- `type`: stable machine-readable error type.
- `message`: short user-safe message.
- `details_event_id`: optional id for matching the error to full logs.

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

Current worker implementation generates mock segment maps and masks out fully
transparent pixels using `alpha_mask_path`. Real SAM2 inference will reuse the
same protocol and replace mock generation inside the worker.
