# Architecture

## Runtime Principles

Inference не должен блокировать UI Krita.

- UI plugin and canvas integration stay responsive.
- Model download, runtime bootstrap and inference run in a worker process.
- Worker sends progress, cancellation, result and error events.
- PyTorch, CUDA, Hugging Face custom model code or dependency failures must not
  crash Krita.

## C++ / Python Split

C++ часть отвечает за:

- native Krita tool lifecycle;
- canvas/image coordinates;
- native lasso/freehand input;
- selection modifier behavior;
- flattened RGB snapshot export;
- document alpha and lasso region mask export;
- applying returned `selection_mask.npy` as a Krita selection;
- stale-result protection when tool, image or document changes.

Python часть отвечает за:

- loader wrapper for the native library;
- worker process entrypoint;
- BiRefNet model registry;
- model download/cache verification;
- managed runtime bootstrap;
- BiRefNet alpha inference;
- clipping alpha to `lasso_region & document_alpha`.

The boundary is intentionally small: C++ writes temp files and sends paths,
worker returns one `selection_mask.npy` plus metadata, C++ reads the mask into
memory and applies it.

## Canvas Integration

The canvas/tool layer is native C++.

- Activating Smart Select does not start inference.
- The user draws a lasso region.
- During worker execution Tool Options shows the current stage and cancel.
- Result is applied automatically as a native Krita selection.
- There is no full-image hover/click segment selection in the MVP path.

## Selection Behavior

Smart Select should follow native Krita selection conventions:

- default lasso completion replaces selection;
- Shift adds;
- Alt/Ctrl actions follow Krita `SelectionAction`;
- click outside current selection should deselect through native Krita selection
  helper semantics when possible;
- Grow, Feather and Anti-aliasing are standard post-selection options.

## Preparing UI Stages

Current worker stages:

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

If the user switches tool, closes the image or cancels, C++ ignores stale late
results.

## Packaging Target

- Linux.
- Krita 5.3.1.
- Native C++ module/library.
- Python extension as loader wrapper.

Cross-version Krita compatibility is not a first-MVP goal.

## Current Implementation Status

Implemented:

- native Smart Select tool registration;
- native lasso/freehand Smart Select input;
- click-outside deselect behavior through Krita selection helper;
- flattened RGB snapshot and alpha mask export;
- lasso region mask export;
- worker-backed `prepare_selection` through `QProcess`;
- managed BiRefNet runtime venv/cache;
- pinned BiRefNet HR Matting model registry and Hugging Face snapshot cache;
- soft `uint8` alpha selection mask result;
- C++ soft alpha preservation in Krita selection apply;
- Tool Options progress/cancel/status;
- OTel-compatible structured logs;
- opt-in debug export for BiRefNet alpha;
- typed Python worker tests and C++ core tests.

Not implemented:

- model picker;
- alternative model backends;
- prompt point editing;
- preview/confirm flow;
- persistent logs and support dialog;
- GUI automation.
