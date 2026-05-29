# Krita Smart Select

Krita Smart Select is a Linux plugin for Krita 5.3.1 that creates soft
AI-assisted selections from a user-drawn region.

The tool works like a native lasso selection tool first: draw or adjust the
area around an object, then press `Select object with AI`. The worker runs
BiRefNet HR Matting on the selected crop and applies the returned soft alpha as
a Krita selection.

## Features

- Native Krita canvas tool and Tool Options integration.
- Replace/add/subtract/intersect selection actions through Krita selection
  semantics.
- Soft alpha selection masks instead of binary-only selections.
- Managed user-level Python runtime for inference dependencies.
- Pinned BiRefNet HR Matting model snapshot cache.
- Local structured logs with OpenTelemetry-compatible field names.
- Debug image export for local diagnosis when explicitly enabled.

## Requirements

For users:

- Linux.
- Krita 5.3.1.
- `uv` available in `PATH`.
- Network access for the first runtime and model download.
- NVIDIA driver with CUDA support for GPU inference, or CPU fallback when CUDA
  is unavailable.
- Enough free disk space for the managed runtime and model cache.

For development:

- CMake.
- C++20 compiler.
- Krita 5.3.1 source tree and build tree.
- Qt5/KF5/Krita development dependencies required by that Krita build.
- Python with `uv`.
- `clang-format`.

The plugin does not install PyTorch, model dependencies or model checkpoints
inside the plugin source tree. Runtime data is stored under Krita user data:

```text
<krita-app-data>/smart_select/
  models/
  runtime/
```

## User Installation

Build or download a release archive with this layout:

```text
krita-smart-select/
  pykrita/
    smart_select.desktop
    smart_select/
      lib/kritasmartselect.so
      ...
  share/krita/actions/SmartSelectTool.action
```

Install into the current user Krita profile:

```bash
mkdir -p ~/.local/share/krita/pykrita ~/.local/share/krita/actions
rm -rf ~/.local/share/krita/pykrita/smart_select
cp -a pykrita/smart_select ~/.local/share/krita/pykrita/
cp pykrita/smart_select.desktop ~/.local/share/krita/pykrita/
cp share/krita/actions/SmartSelectTool.action ~/.local/share/krita/actions/
```

Restart Krita, enable the plugin if needed, then select `Smart Select` from the
toolbox.

On first use, draw or edit a region, press `Select object with AI`, and let the
worker create the runtime and download the model.

## Development Setup

Create the development Python environment:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

Build against a Krita 5.3.1 source/build tree:

```bash
KRITA_SRC=/home/bogdan/Work/sources/krita \
KRITA_BUILD=/home/bogdan/Work/build/krita-5.3.1 \
BUILD_JOBS=20 \
./scripts/build-krita-5.3.1.sh
```

The build output is written to:

```text
dist/krita-smart-select/
```

Install the local build into Krita:

```bash
mkdir -p ~/.local/share/krita/pykrita ~/.local/share/krita/actions
rm -rf ~/.local/share/krita/pykrita/smart_select
cp -a dist/krita-smart-select/pykrita/smart_select ~/.local/share/krita/pykrita/
cp dist/krita-smart-select/pykrita/smart_select.desktop ~/.local/share/krita/pykrita/
cp dist/krita-smart-select/share/krita/actions/SmartSelectTool.action ~/.local/share/krita/actions/
```

Package a Linux archive:

```bash
./scripts/package-linux.sh
```

## Verification

Run the default checks:

```bash
PYTHON=.venv/bin/python ./scripts/test-code.sh
```

This runs Python compile checks, Ruff, mypy, worker unit tests and the C++ core
tests when the Krita source/build tree is available.

Run the opt-in real inference smoke test only when model/runtime setup is
acceptable:

```bash
SMART_SELECT_RUN_BIREFNET_SMOKE=1 PYTHON=.venv/bin/python ./scripts/test-birefnet-smoke.sh
```

## Documentation

- [Product Scope](docs/product-scope.md)
- [Architecture](docs/architecture.md)
- [Worker Protocol](docs/worker-protocol.md)
- [Snapshot And Temporary Files](docs/snapshot-and-temp-files.md)
- [Models](docs/models.md)
- [Runtime Environment](docs/runtime-env.md)
- [Observability](docs/engineering/observability.md)
- [Typing](docs/engineering/typing.md)
- [Testing](docs/engineering/testing.md)
- [Code Style](docs/engineering/code-style.md)
- [Licensing](docs/licensing.md)
- [Contributing](CONTRIBUTING.md)

## Acknowledgements

Krita Smart Select builds on ideas, APIs and open-source work from:

- [Krita](https://krita.org/) and the KDE Krita project, whose native tool,
  selection and plugin APIs make this kind of canvas-integrated workflow
  possible.
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet), especially the
  `ZhengPeng7/BiRefNet_HR-matting` model used for direct soft-alpha
  extraction.
- [Krita Vision Tools](https://github.com/Acly/krita-ai-tools), which provided
  a useful reference for ML-assisted Krita tooling and user-facing selection
  workflows.
- [CorridorKey](https://github.com/nikopueringer/CorridorKey), which helped
  inform the product direction toward alpha-first, plug-and-play matting.

## License

Krita Smart Select is licensed under GPL-3.0-only. See [LICENSE](LICENSE)
and [docs/licensing.md](docs/licensing.md).
