# Snapshot And Temporary Files

## Snapshot Inputs

C++ exports these files for each AI selection job:

- flattened RGB PNG snapshot for model input;
- document alpha mask as NumPy `.npy` with dtype `|u1`;
- current selection region mask as NumPy `.npy` with dtype `|u1`.

The current selection region is a hard safety boundary. Worker output is
clipped to the selected region and document alpha before returning.

## Fingerprint

The fingerprint describes the exported image and selected region for
stale-result checks and future result-cache keys.

Current algorithm version:

```text
birefnet-hr-matting-lasso-v1
```

Changing the model id or algorithm version must invalidate any future persisted
selection result cache.

## Cleanup

Temporary snapshot/result files are deleted after C++ reads a successful result
or cancels a job.

Set `SMART_SELECT_KEEP_DEBUG_FILES=1` to keep debug exports:

```bash
./scripts/clean-debug-data.sh
SMART_SELECT_KEEP_DEBUG_FILES=1 /path/to/krita.AppImage
```

Debug exports may contain document-derived image data and should be shared only
when the user explicitly wants to share that image.

## Runtime And Model Caches

Runtime and model caches are persistent user data:

- `smart_select/runtime` for the managed venv;
- `smart_select/models/birefnet_hr_matting` for the pinned model snapshot.

The worker may recreate the managed runtime when its manifest version changes.
It does not delete verified model snapshots automatically.
