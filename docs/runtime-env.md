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

- runtime version: `sam2-runtime-v3`
- runtime Python: `3.12`
- SAM2 source: `facebookresearch/sam2`
- SAM2 source ref: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- PyTorch wheel index: `https://download.pytorch.org/whl/cu124`
- SAM2 custom CUDA extension: enabled with `SAM2_BUILD_CUDA=1`
- SAM2 extension build errors: fail-fast with `SAM2_BUILD_ALLOW_ERRORS=0`

Install sequence:

```bash
<venv>/bin/python -m pip install --upgrade pip setuptools wheel
<venv>/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu124 "torch>=2.5.1" "torchvision>=0.20.1"
<venv>/bin/python -m pip install "numpy>=1.26,<3" "pillow>=10,<13"
SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0 <venv>/bin/python -m pip install --force-reinstall --no-deps --no-build-isolation "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
```

SAM2 is installed without build isolation so the CUDA extension is compiled
against the runtime's pinned PyTorch CUDA wheel, not a temporary build
environment with a different Torch/CUDA version.

The worker validates:

- worker and runtime Python are compatible with the pinned runtime Python;
- `torch` import;
- `PIL.Image` import;
- `sam2.build_sam` import;
- `sam2.automatic_mask_generator` import;
- `sam2._C` import;
- torch version `>=2.5.1`.

CUDA availability is reported, but not required for validation.

For CUDA extension builds the worker looks for a CUDA toolkit in this order:

1. `SMART_SELECT_CUDA_HOME`;
2. `CUDA_HOME`;
3. `CUDA_PATH`;
4. `/usr/local/cuda-12.4`;
5. `/usr/local/cuda`;
6. `~/.local/share/krita/smart_select/cuda-12.4`;
7. `~/miniconda3`;
8. `nvcc` in `PATH`.

The toolkit should match the PyTorch CUDA wheel family. The current runtime uses
PyTorch `cu124`, so CUDA toolkit `12.4.x` is the intended build toolkit.

If the selected CUDA prefix contains conda compiler wrappers
`x86_64-conda-linux-gnu-gcc` and `x86_64-conda-linux-gnu-g++`, the worker uses
them as `CC`, `CXX` and `CUDAHOSTCXX` for SAM2 extension builds. This avoids
accidentally compiling CUDA 12.4 code with an unsupported system compiler.

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

The launcher also applies the selected CUDA toolkit path to the child worker
process before Python 3.12 starts. This is required for loading compiled CUDA
extensions such as `sam2._C`, because dynamic library lookup needs the CUDA
library directories at process startup.

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
cache state, CUDA toolkit path and `sam2._C` availability, but never installs
dependencies and never reads image inputs.
