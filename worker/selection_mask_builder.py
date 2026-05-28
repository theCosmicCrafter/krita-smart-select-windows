from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

from observability import log_event
from region_selection import RegionSelectionInput

MIN_INSIDE_AREA = 32
MIN_REGION_COVERAGE = 0.05


class EmptySelectionError(ValueError):
    pass


class SegmentMetadata(TypedDict):
    id: int
    bbox: list[int]
    area: int
    score: NotRequired[float]


@dataclass(frozen=True)
class SelectionMaskBuildResult:
    selection_mask: bytes
    metadata: list[SegmentMetadata]
    fallback_used: bool


def build_selection_mask(
    masks: Sequence[Any], scores: Sequence[float], region: RegionSelectionInput
) -> SelectionMaskBuildResult:
    crop_width = region.crop_bbox[2]
    crop_height = region.crop_bbox[3]
    region_area = _count_region(region.crop_region_mask, region.crop_alpha_mask)
    if region_area <= 0:
        raise EmptySelectionError("selection is empty")

    best_index = -1
    best_score = -1.0
    best_mask: list[bool] | None = None
    for index, mask in enumerate(masks):
        candidate = _mask_to_bool_list(mask, crop_width, crop_height)
        stats = _candidate_stats(candidate, region.crop_region_mask, region.crop_alpha_mask)
        if stats.inside_area < MIN_INSIDE_AREA:
            continue
        if stats.region_coverage < MIN_REGION_COVERAGE:
            continue
        model_score = float(scores[index]) if index < len(scores) else 0.0
        rank = 0.45 * model_score + 0.35 * stats.region_coverage + 0.20 * stats.candidate_coverage
        if rank > best_score:
            best_index = index
            best_score = rank
            best_mask = candidate

    fallback_used = best_mask is None
    if fallback_used:
        log_event(
            "selection_mask.fallback",
            severity="warning",
            component="worker",
            **{"error.reason": "no_candidate"},
        )
        best_mask = [value > 0 for value in region.crop_region_mask]
        best_score = 1.0
        best_index = -1

    assert best_mask is not None
    crop_final = bytearray(crop_width * crop_height)
    for index, selected in enumerate(best_mask):
        if selected and region.crop_region_mask[index] > 0 and region.crop_alpha_mask[index] > 0:
            crop_final[index] = 255

    full = bytearray(region.canvas_width * region.canvas_height)
    crop_x, crop_y, _, _ = region.crop_bbox
    min_x = region.canvas_width
    min_y = region.canvas_height
    max_x = -1
    max_y = -1
    area = 0
    for y in range(crop_height):
        for x in range(crop_width):
            if crop_final[y * crop_width + x] == 0:
                continue
            canvas_x = crop_x + x
            canvas_y = crop_y + y
            full[canvas_y * region.canvas_width + canvas_x] = 255
            min_x = min(min_x, canvas_x)
            min_y = min(min_y, canvas_y)
            max_x = max(max_x, canvas_x)
            max_y = max(max_y, canvas_y)
            area += 1

    if area <= 0:
        raise EmptySelectionError("selection is empty")

    metadata: list[SegmentMetadata] = [
        {
            "id": 1,
            "bbox": [min_x, min_y, max_x - min_x + 1, max_y - min_y + 1],
            "area": area,
            "score": best_score if best_index >= 0 else 1.0,
        }
    ]
    return SelectionMaskBuildResult(
        selection_mask=bytes(full), metadata=metadata, fallback_used=fallback_used
    )


@dataclass(frozen=True)
class _CandidateStats:
    inside_area: int
    candidate_coverage: float
    region_coverage: float


def _candidate_stats(
    candidate: Sequence[bool], region_mask: bytes, alpha_mask: bytes
) -> _CandidateStats:
    mask_area = 0
    inside_area = 0
    region_area = _count_region(region_mask, alpha_mask)
    for index, selected in enumerate(candidate):
        if not selected or alpha_mask[index] == 0:
            continue
        mask_area += 1
        if region_mask[index] > 0:
            inside_area += 1
    return _CandidateStats(
        inside_area=inside_area,
        candidate_coverage=inside_area / mask_area if mask_area else 0.0,
        region_coverage=inside_area / region_area if region_area else 0.0,
    )


def _count_region(region_mask: bytes, alpha_mask: bytes) -> int:
    return sum(1 for index, value in enumerate(region_mask) if value > 0 and alpha_mask[index] > 0)


def _mask_to_bool_list(mask: Any, width: int, height: int) -> list[bool]:
    expected = width * height
    values: list[bool] = []
    for y in range(height):
        for x in range(width):
            values.append(_mask_value(mask, x, y, y * width + x))
    if len(values) != expected:
        raise ValueError("Mask size is invalid")
    return values


def _mask_value(mask: Any, x: int, y: int, flat_index: int) -> bool:
    try:
        return bool(mask[y, x])
    except Exception:
        pass
    try:
        row = mask[y]
        return bool(row[x])
    except Exception:
        pass
    try:
        return bool(mask[flat_index])
    except Exception:
        return False
