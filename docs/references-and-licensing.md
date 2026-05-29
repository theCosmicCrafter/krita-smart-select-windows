# References And Licensing

## Reference Projects

Project implementation may use ideas from:

- Smart Segments
- Krita Vision Tools

## Krita Vision Tools Reference Value

Krita Vision Tools is useful as a reference for:

- C++ plugin/module build under Krita;
- native tool registration;
- canvas/image coordinates;
- pixel selection apply through Krita APIs;
- overlay drawing through `paint()` and `updateCanvas()`;
- Python loader wrapper for native library.
- native lasso/selection tool behavior where applicable.

## What Krita Vision Tools Does Not Answer

Krita Vision Tools does not answer these Krita Smart Select decisions:

- IPC between C++ tool and separate Python/PyTorch worker;
- quality tradeoffs for lasso/region-prompted SAM2 selection;
- `selection_mask` / `segment_map` result contracts;
- candidate mask ranking and debug export;
- segment map/mask cache;
- SAM2 PyTorch dependency management.

## Smart Segments Reference Value

Smart Segments is useful as a reference for:

- SAM2/PyTorch pipeline;
- converting model output into masks/segments;
- connecting segmentation results back to Krita pixels/selections.

InvokeAI is useful as a product/UX reference for prompt-driven SAM selection:
box/point/prompt inputs tend to produce more controllable masks than one
full-image automatic segmentation pass. Krita Smart Select uses that direction
for the MVP lasso workflow, while keeping implementation native to Krita.

Matting-Anything is the current alpha refinement reference:

- project: `SHI-Labs/Matting-Anything`;
- source ref: `eb01e1c1f0016e2a4d00615b115f3b5ddcd031e4`;
- product path: MAM SAM ViT-H (`mam_sam_vith`);
- refiner role: consume the recall-oriented SAM2 prior and return soft alpha.

The plugin must preserve attribution for Matting-Anything and its bundled
Segment Anything dependency before public release. MAM is not vendored into the
plugin source tree; it is cloned into the managed runtime source cache.

## Attribution

The project must include correct license notices and attribution for any reused
code or substantial implementation patterns.

## License Direction

Project is expected to be GPLv3-compatible. This matches practical licensing
expectations for Krita plugin development and keeps compatibility with patterns
from Krita Vision Tools where needed.

Exact license files and third-party notices must be finalized before the first
public release.
