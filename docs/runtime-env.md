# Runtime Environment

## Decision

Real SAM2 inference uses a user-level runtime environment managed by the
worker. The plugin does not vendor PyTorch/SAM2 and does not install runtime
packages into the plugin source tree.

Runtime layout:

```text
<krita-app-data>/smart_select/
  runtime/
    venv/
    pip-cache/
    runtime.json
  models/
    sam2.1_hiera_large/
      sam2.1_hiera_large.pt
      model.json
```

If Krita does not provide `QStandardPaths::AppDataLocation`, the fallback is:

```text
~/.local/share/krita-smart-select/
```

## Bootstrap

The C++ tool passes `runtime_cache_dir` to the worker. The worker creates and
validates the runtime automatically.

Pinned runtime defaults:

- runtime version: `sam2-runtime-v2`
- runtime Python: `3.12`
- SAM2 source: `facebookresearch/sam2`
- SAM2 source ref: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- PyTorch wheel index: `https://download.pytorch.org/whl/cu124`
- SAM2 custom CUDA extension: disabled with `SAM2_BUILD_CUDA=0`

Install sequence:

```bash
<venv>/bin/python -m pip install --upgrade pip setuptools wheel
<venv>/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu124 "torch>=2.5.1" "torchvision>=0.20.1"
SAM2_BUILD_CUDA=0 <venv>/bin/python -m pip install "numpy>=1.26,<3" "pillow>=10,<13" "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
```

The worker validates:

- worker and runtime Python are compatible with the pinned runtime Python;
- `torch` import;
- `PIL.Image` import;
- `sam2.build_sam` import;
- `sam2.automatic_mask_generator` import;
- torch version `>=2.5.1`.

CUDA availability is reported, but not required for validation.

If the system Python can create a venv but the venv has no `pip` because
`ensurepip` is unavailable, the worker falls back to `uv pip install --python
<venv-python>`. The fallback still installs into the same managed venv and uses
the runtime cache directory via `UV_CACHE_DIR`.

The Krita Python loader starts the worker through `worker_launcher.py` when
`uv` is available. The launcher runs the worker with the pinned runtime Python
instead of the host `python3`, which avoids trying to install PyTorch into
unsupported interpreter versions such as Python 3.14. If the cached venv was
created with a different Python major/minor version, the worker treats it as
invalid and recreates it.

## Inference Device Policy

- Use CUDA when `torch.cuda.is_available()`.
- Use CPU only when CUDA is unavailable.
- CUDA out-of-memory is reported as `InferenceOutOfMemory`.
- CUDA OOM is not silently retried on CPU.

## Cancellation

The worker runs `prepare_segments` jobs in a background thread so stdin remains
available for `cancel` requests.

Cancellation can stop:

- runtime bootstrap between subprocess calls;
- active pip subprocesses;
- model downloads between chunks.

SAM inference itself is not forcibly interrupted mid-GPU-call. If cancellation
arrives during inference, the worker checks it immediately after the call,
cleans temp output and does not emit a result.

## Diagnostics

The `diagnostics` request may include `runtime_cache_dir`. It reports runtime
cache state and importability, but never installs dependencies and never reads
image inputs.
