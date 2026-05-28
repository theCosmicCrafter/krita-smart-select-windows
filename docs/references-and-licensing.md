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

## What Krita Vision Tools Does Not Answer

Krita Vision Tools does not answer these Krita Smart Select decisions:

- IPC between C++ tool and separate Python/PyTorch worker;
- prepared automatic segmentation of the whole image;
- `segment_map` / `pixel -> segment_id` format;
- hover highlighting of segments in segmentation tool;
- segment map/mask cache;
- SAM2 PyTorch dependency management.

## Smart Segments Reference Value

Smart Segments is useful as a reference for:

- SAM2/PyTorch pipeline;
- converting model output into masks/segments;
- connecting segmentation results back to Krita pixels/selections.

## Attribution

The project must include correct license notices and attribution for any reused
code or substantial implementation patterns.

## License Direction

Project is expected to be GPLv3-compatible. This matches practical licensing
expectations for Krita plugin development and keeps compatibility with patterns
from Krita Vision Tools where needed.

Exact license files and third-party notices must be finalized before the first
public release.
