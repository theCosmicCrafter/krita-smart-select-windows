# Architecture

## Runtime Principles

Inference must not block the Krita UI.

- UI plugin and canvas integration stay responsive.
- Model download, runtime bootstrap and inference run in a worker process.
- Worker sends progress, cancellation, result and error events.
- PyTorch, CUDA, Hugging Face custom model code or dependency failures must not
  crash Krita.

## C++ / Python Split

The C++ layer is responsible for:

- native Krita tool lifecycle;
- canvas/image coordinates;
- native lasso/freehand input;
- selection modifier behavior;
- flattened RGB snapshot export;
- document alpha and selected-region mask export;
- applying returned `selection_mask.npy` as a Krita selection;
- stale-result protection when tool, image or document changes.

The Python layer is responsible for:

- loader wrapper for the native library;
- worker process entrypoint;
- BiRefNet model registry;
- model download/cache verification;
- managed runtime bootstrap;
- BiRefNet alpha inference;
- clipping alpha to `selection_region & document_alpha`.

The boundary is intentionally small: C++ writes temp files and sends paths,
worker returns one `selection_mask.npy` plus metadata, C++ reads the mask into
memory and applies it.

## Canvas Integration

The canvas/tool layer is native C++.

- Activating Smart Select does not start inference.
- Lasso completion edits the current Krita selection using native
  replace/add/subtract/intersect behavior.
- The worker starts only when the user presses `Select object with AI`; C++
  exports the current selection as the region mask.
- During worker execution Tool Options shows the current stage and cancel.
- Result is applied automatically as a native Krita selection.

## Selection Behavior

Smart Select should follow native Krita selection conventions:

- default lasso completion replaces selection;
- Shift adds;
- Alt/Ctrl actions follow Krita `SelectionAction`;
- `Select object with AI` refines the current selection, so users can edit the
  region and rerun without switching tools;
- click outside current selection should clear the selection through native
  Krita selection helper semantics when possible;
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

## Platform Target

- Linux.
- Krita 5.3.1.
- Native C++ module/library.
- Python extension as loader wrapper.

Cross-version Krita compatibility is handled as an explicit release decision.
