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

The venv is created with `python -m venv`, so its layout follows the OS: the
interpreter is `venv/bin/python` on Linux and `venv\Scripts\python.exe` on
Windows (`runtime_env.venv_python`), and that scripts directory (`bin` or
`Scripts`) is prepended to `PATH` when the runtime is activated. The site-packages
path is not assumed: it is read from the venv interpreter via `sysconfig`.

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

`runtime.json` also records the `torch_index` the venv was built from (a missing
field means the default `cu124` index). The worker recreates the venv when the index
it wants now differs from the recorded one, see [Torch Wheel Index](#torch-wheel-index).

## Dependencies

Runtime installs:

- PyTorch/torchvision from a CUDA wheel index, `cu124` by default and `cu128` for
  Blackwell GPUs (both serve `cp312` wheels for Linux and Windows `win_amd64`; the
  same version ranges resolve on both), see [Torch Wheel Index](#torch-wheel-index);
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

## Torch Wheel Index

The `cu124` wheels contain no kernels for Blackwell GPUs (compute capability 10.0
and up, e.g. GeForce RTX 50xx, which is `sm_120`): torch reports the GPU as
available but cannot run on it. Before the venv is created the worker therefore
asks the driver for the GPU generation:

```text
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

- highest capability `>= 10.0`: `https://download.pytorch.org/whl/cu128`;
- any other NVIDIA GPU: `https://download.pytorch.org/whl/cu124` (unchanged);
- `nvidia-smi` missing, failing (old drivers do not know `compute_cap`) or timing
  out: the capability is unknown. A new runtime uses `cu124`; an existing runtime
  keeps the index it was built with, so a failed probe never causes a rebuild of a
  multi-GB runtime.

The chosen index is stored in `runtime.json` as `torch_index`. If an existing
runtime was built from a different index than the one now wanted (new GPU, or a
runtime created before Blackwell support, which counts as `cu124`) the venv is
recreated; models are kept. Both indexes are in `worker/runtime_manifest.py`. The
runtime stays under Krita user data, never in the plugin tree.

The probe and the choice are logged as `runtime.torch_index.selected` and
`runtime.torch_index.changed`.

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

Before running on CUDA the worker compares the GPU capability with
`torch.cuda.get_arch_list()` using torch's own rule (architectures are compatible
within a major version). If the installed torch has no kernels for the GPU it logs
`alpha_model.cuda.incompatible` (device name, compute capability, torch version, arch
list, `error.reason=cuda_arch_not_supported`) and returns
`InferenceUnsupportedGpu` with a message naming the GPU, the capability and the
architectures torch was built for. It does not fall back to the CPU: the user
should fix the runtime, not silently wait minutes per selection. A supported GPU is
logged as `alpha_model.cuda.checked`. Other inference failures return
`alpha extraction failed (<device>, <exception class>)` and log
`alpha_model.predict.failed` with the exception message (never image data).

## GPU Memory Policy

The worker keeps the on-disk runtime and model snapshot caches, but unloads the
CUDA BiRefNet model and clears the PyTorch CUDA allocator cache after each
lasso job. This avoids VRAM growth inside Krita after repeated selections. The
tradeoff is that the next selection reloads the model from the verified local
snapshot.

## Cancellation

`cancel` stops the running runtime setup subprocess (venv, pip) and the job ends with
`RuntimeBootstrapError: runtime setup cancelled`. On POSIX the subprocess is
terminated. On Windows the whole process tree is stopped with `taskkill /PID <pid> /T
/F` (and `Popen.kill` if `taskkill` is unavailable), because the venv `python.exe` is
a launcher stub that runs the real interpreter, and so can other tools, as a child;
the kill is logged as `worker.job.process.terminated` (pid, method, result). A cancel
that arrives between starting a subprocess and registering it is applied as soon as
the subprocess is registered.

Helper subprocesses (venv, pip, `nvidia-smi`, `taskkill`) are started with
`stdin=DEVNULL`. They never need the NDJSON protocol stdin, and on Windows a Python
child that inherits it can block at startup while the worker's main thread is waiting
for the next request on that pipe.

The model download runs in-process through `huggingface_hub.snapshot_download` and
is not interrupted by `cancel`; the cancellation is noticed when it returns.

## Worker Launch

C++ starts the worker with `QProcess` as `<program> <worker script>`; the
program is chosen by `python/smart_select/extension.py`.

- Linux: `worker_launcher.py` when `uv` is in `PATH`, which `exec`s
  `uv run --no-project --python 3.12 python <script>` so the worker runs on the
  runtime Python version. Without `uv` it falls back to `python3`.
- Windows: `worker_launcher.cmd`, which runs the same `uv run` command with
  inherited stdio and `PYTHONUTF8=1`. `QProcess` cannot start a `.py` file there,
  Krita ships no `python.exe`, and `os.exec*` would exit the launcher
  immediately. `uv` is required; no system Python is.

The Python version in the launchers must match `RUNTIME_PYTHON_VERSION` in
`worker/runtime_manifest.py`.

## Security

BiRefNet uses Hugging Face custom code through `trust_remote_code=True`. This is
allowed only for the pinned curated repo/revision in `docs/models.md`.
User-provided model repos are out of scope.
