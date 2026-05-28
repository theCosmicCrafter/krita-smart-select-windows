from __future__ import annotations

import sys
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict, TypeGuard

MAX_UINT16_SEGMENTS = 65_535
MIN_SEGMENT_AREA = 32


class SegmentMetadata(TypedDict):
    id: int
    bbox: list[int]
    area: int
    score: NotRequired[float]


@dataclass(frozen=True)
class SegmentMapBuildResult:
    segment_map: array[int]
    metadata: list[SegmentMetadata]
    truncated: bool


def build_segment_map_from_masks(
    masks: Sequence[Mapping[str, object]],
    *,
    canvas_width: int,
    canvas_height: int,
    inference_width: int,
    inference_height: int,
    alpha_mask: bytes,
) -> SegmentMapBuildResult:
    if len(alpha_mask) != inference_width * inference_height:
        raise ValueError("Alpha mask size does not match inference size")

    segment_map = array("H", [0]) * (canvas_width * canvas_height)
    metadata: list[SegmentMetadata] = []
    truncated = False

    for mask in sorted(masks, key=_mask_sort_key):
        if len(metadata) >= MAX_UINT16_SEGMENTS:
            truncated = True
            break

        segmentation = mask.get("segmentation")
        if segmentation is None:
            continue

        source_bbox = _mask_bbox(mask, inference_width, inference_height)
        if source_bbox is None:
            continue

        segment_id = len(metadata) + 1
        x0, y0, x1, y1 = _scale_bbox_to_canvas(
            source_bbox, canvas_width, canvas_height, inference_width, inference_height
        )

        min_x = canvas_width
        min_y = canvas_height
        max_x = -1
        max_y = -1
        area = 0
        for y in range(y0, y1):
            source_y = _scale_index(y, canvas_height, inference_height)
            for x in range(x0, x1):
                output_index = y * canvas_width + x
                if segment_map[output_index] != 0:
                    continue

                source_x = _scale_index(x, canvas_width, inference_width)
                source_index = source_y * inference_width + source_x
                if alpha_mask[source_index] == 0:
                    continue
                if not _mask_value(segmentation, source_x, source_y, source_index):
                    continue

                segment_map[output_index] = segment_id
                area += 1
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

        if area < MIN_SEGMENT_AREA:
            if area > 0:
                _clear_segment(segment_map, segment_id)
            continue

        metadata.append(
            {
                "id": segment_id,
                "bbox": [min_x, min_y, max_x - min_x + 1, max_y - min_y + 1],
                "area": area,
                "score": _mask_score(mask),
            }
        )

    if truncated:
        print(
            "[SmartSelectWorker] event=segment_map.truncated max_segments=65535",
            file=sys.stderr,
            flush=True,
        )

    return SegmentMapBuildResult(segment_map=segment_map, metadata=metadata, truncated=truncated)


def _mask_sort_key(mask: Mapping[str, object]) -> tuple[float, float, int]:
    return (-_number(mask.get("predicted_iou")), -_number(mask.get("stability_score")), _area(mask))


def _mask_score(mask: Mapping[str, object]) -> float:
    predicted_iou = mask.get("predicted_iou")
    if isinstance(predicted_iou, int | float):
        return float(predicted_iou)
    stability_score = mask.get("stability_score")
    if isinstance(stability_score, int | float):
        return float(stability_score)
    return 1.0


def _number(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _area(mask: Mapping[str, object]) -> int:
    value = mask.get("area")
    if isinstance(value, int | float):
        return int(value)
    bbox = mask.get("bbox")
    if _is_bbox(bbox):
        return int(bbox[2]) * int(bbox[3])
    return 0


def _mask_bbox(
    mask: Mapping[str, object], inference_width: int, inference_height: int
) -> tuple[int, int, int, int] | None:
    bbox = mask.get("bbox")
    if _is_bbox(bbox):
        x = max(0, min(inference_width, int(bbox[0])))
        y = max(0, min(inference_height, int(bbox[1])))
        width = max(0, int(bbox[2]))
        height = max(0, int(bbox[3]))
        return (x, y, min(inference_width, x + width), min(inference_height, y + height))
    return (0, 0, inference_width, inference_height)


def _is_bbox(value: object) -> TypeGuard[Sequence[int | float]]:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes | bytearray)
        and len(value) == 4
        and all(isinstance(item, int | float) for item in value)
    )


def _scale_bbox_to_canvas(
    source_bbox: tuple[int, int, int, int],
    canvas_width: int,
    canvas_height: int,
    inference_width: int,
    inference_height: int,
) -> tuple[int, int, int, int]:
    source_x0, source_y0, source_x1, source_y1 = source_bbox
    x0 = max(0, min(canvas_width, source_x0 * canvas_width // inference_width))
    y0 = max(0, min(canvas_height, source_y0 * canvas_height // inference_height))
    x1 = max(
        x0, min(canvas_width, (source_x1 * canvas_width + inference_width - 1) // inference_width)
    )
    y1 = max(
        y0,
        min(canvas_height, (source_y1 * canvas_height + inference_height - 1) // inference_height),
    )
    return x0, y0, x1, y1


def _scale_index(output_index: int, output_size: int, source_size: int) -> int:
    return min(source_size - 1, output_index * source_size // output_size)


def _mask_value(segmentation: Any, x: int, y: int, flat_index: int) -> bool:
    try:
        return bool(segmentation[y, x])
    except Exception:
        pass
    try:
        row = segmentation[y]
        return bool(row[x])
    except Exception:
        pass
    try:
        return bool(segmentation[flat_index])
    except Exception:
        return False


def _clear_segment(segment_map: array[int], segment_id: int) -> None:
    for index, value in enumerate(segment_map):
        if value == segment_id:
            segment_map[index] = 0
