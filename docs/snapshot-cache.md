# Snapshot And Cache

## Snapshot Inputs

C++ exports:

- flattened RGB PNG snapshot for model input;
- document alpha mask as `.npy` `|u1`;
- lasso region mask as `.npy` `|u1`.

The lasso region is a hard safety boundary. Worker output is clipped to the
lasso region and document alpha before returning.

## Fingerprint

Fingerprint is used for stale-result context and future cache keys. Current
selection result disk cache is not implemented.

Current algorithm version:

```text
birefnet-hr-matting-lasso-v1
```

Changing model id or algorithm version should invalidate any future selection
result cache.

## Runtime And Model Cache

Runtime and model caches remain active:

- `smart_select/runtime` for the managed venv;
- `smart_select/models/birefnet_hr_matting` for the pinned HF snapshot.

Old SAM2/MAM caches are unused by the BiRefNet path and may be manually removed
after validation.
