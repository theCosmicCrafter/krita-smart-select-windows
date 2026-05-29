# Product Scope

## MVP Flow

1. User selects Smart Select.
2. User draws a lasso around the target object.
3. C++ exports flattened RGB snapshot, document alpha and lasso region mask.
4. Worker runs BiRefNet HR Matting on a crop around the lasso.
5. Worker clips the soft alpha to `lasso_region & document_alpha`.
6. C++ applies the soft alpha as a native Krita selection.

## In Scope

- Native Krita lasso tool UX.
- Soft alpha selection masks.
- BiRefNet managed runtime and model cache.
- Cancel/stale-result safety.
- OTel-compatible local logs.
- Debug exports for alpha diagnosis.

## Out Of Scope

- SAM2/MAM/trimap path.
- Full-image prepared hover/click segmentation.
- `mock` model id.
- Model picker.
- Positive/negative point UI.
- Preview/confirm flow.
- Disk cache for selection results.
- GUI automation.
- Network telemetry.

## Selection Options

Krita Grow, Feather and Anti-aliasing stay visible because they are native
post-selection options. The worker does not add hidden grow or feather.
