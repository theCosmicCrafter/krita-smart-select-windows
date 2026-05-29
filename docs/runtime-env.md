# Runtime Environment

Smart Select uses a managed user-level runtime for BiRefNet inference. The
plugin source tree must remain free of runtime installs and model checkpoints.

## Cache Layout

Primary location comes from Krita user data via C++ `QStandardPaths`:

```text
<krita-app-data>/smart_select/
  models/
    birefnet_hr_matting/
      snapshot/
        model.safetensors
        config.json
        *.py
      model.json
  runtime/
    venv/
    pip-cache/
    runtime.json
```

Fallback:

```text
~/.local/share/krita-smart-select/
  models/
  runtime/
```

## Runtime Version

Current runtime:

```text
birefnet-runtime-v1
```

If `runtime.json` has a different runtime version, the worker recreates the
managed venv. It does not delete model caches.

## Dependencies

Runtime installs:

- PyTorch/torchvision from CUDA 12.4 wheel index;
- `numpy`;
- `pillow`;
- `transformers`;
- `huggingface-hub`;
- `safetensors`;
- `timm`;
- `kornia`;
- `einops`;
- `scipy`;
- `scikit-image`;
- `accelerate`.

## Validation

Validation imports:

- `torch`;
- `PIL.Image`;
- `torchvision.transforms`;
- `transformers.AutoModelForImageSegmentation`;
- `huggingface_hub.snapshot_download`;
- `safetensors`.

CUDA availability is reported but not required. Device policy is CUDA first,
CPU only when CUDA is unavailable, and CUDA OOM is returned as a user-safe error
without silent CPU retry.

## GPU Memory Policy

The worker keeps the on-disk runtime and model snapshot caches, but unloads the
CUDA BiRefNet model and clears the PyTorch CUDA allocator cache after each
lasso job. This avoids VRAM growth inside Krita after repeated selections. The
tradeoff is that the next selection reloads the model from the verified local
snapshot.

## Security

BiRefNet uses Hugging Face custom code through `trust_remote_code=True`. This is
allowed only for the pinned curated repo/revision in `docs/models.md`.
User-provided model repos are out of scope.
