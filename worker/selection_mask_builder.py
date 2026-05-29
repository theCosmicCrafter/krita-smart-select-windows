from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

from observability import log_event
from region_selection import RegionSelectionInput

MIN_INSIDE_AREA = 32
MIN_REGION_COVERAGE = 0.05
MIN_RECALL_POSITIVE_HIT_RATE = 0.50
MAX_RECALL_NEGATIVE_HIT_RATE = 0.25
MAX_HOLE_FILL_AREA = 4096
MAX_HOLE_FILL_REGION_FRACTION = 0.08


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


@dataclass(frozen=True)
class SamPriorBuildResult:
    crop_prior_mask: bytes
    fallback_used: bool
    merged_count: int
    best_score: float
    region_area: int
    prior_area: int

    @property
    def region_coverage(self) -> float:
        return self.prior_area / self.region_area if self.region_area else 0.0


@dataclass(frozen=True)
class _RankedCandidate:
    index: int
    mask: list[bool]
    stats: _CandidateStats
    score: float
    rank: float


def build_selection_mask(
    masks: Sequence[Any], scores: Sequence[float], region: RegionSelectionInput
) -> SelectionMaskBuildResult:
    prior = build_sam_prior(masks, scores, region)
    return build_selection_mask_from_crop_alpha(
        prior.crop_prior_mask,
        region,
        prior.best_score,
        fallback_used=prior.fallback_used,
    )


def build_sam_prior(
    masks: Sequence[Any], scores: Sequence[float], region: RegionSelectionInput
) -> SamPriorBuildResult:
    crop_width = region.crop_bbox[2]
    crop_height = region.crop_bbox[3]
    region_area = _count_region(region.crop_region_mask, region.crop_alpha_mask)
    if region_area <= 0:
        raise EmptySelectionError("selection is empty")

    candidates: list[_RankedCandidate] = []
    for index, mask in enumerate(masks):
        candidate = _mask_to_bool_list(mask, crop_width, crop_height)
        stats = _candidate_stats(candidate, region)
        if stats.inside_area < MIN_INSIDE_AREA:
            continue
        if stats.region_coverage < MIN_REGION_COVERAGE:
            continue
        model_score = float(scores[index]) if index < len(scores) else 0.0
        rank = (
            0.35 * model_score
            + 0.25 * stats.region_coverage
            + 0.15 * stats.candidate_coverage
            + 0.20 * stats.positive_hit_rate
            - 0.25 * stats.negative_hit_rate
            - 0.20 * stats.boundary_coverage
        )
        candidates.append(
            _RankedCandidate(
                index=index,
                mask=candidate,
                stats=stats,
                score=model_score,
                rank=rank,
            )
        )

    fallback_used = not candidates
    if fallback_used:
        log_event(
            "selection_mask.fallback",
            severity="warning",
            component="worker",
            **{"error.reason": "no_candidate"},
        )
        best_mask = [value > 0 for value in region.crop_region_mask]
        best_score = 1.0
        merged_count = 0
    else:
        best = max(candidates, key=lambda candidate: candidate.rank)
        best_mask, merged_count = _build_recall_prior(best, candidates, region)
        best_score = best.rank
        log_event(
            "selection_mask.prior.result",
            component="worker",
            **{
                "masks.count": len(candidates),
                "selection.prior.merged_count": merged_count,
                "selection.prior.best_rank": round(best.rank, 4),
                "selection.prior.best_region_coverage": round(best.stats.region_coverage, 4),
            },
        )

    best_mask = _fill_small_holes(
        best_mask, region.crop_region_mask, region.crop_alpha_mask, crop_width, crop_height
    )
    crop_prior = bytearray(crop_width * crop_height)
    prior_area = 0
    for index, selected in enumerate(best_mask):
        if selected and region.crop_region_mask[index] > 0 and region.crop_alpha_mask[index] > 0:
            crop_prior[index] = 255
            prior_area += 1

    return SamPriorBuildResult(
        crop_prior_mask=bytes(crop_prior),
        fallback_used=fallback_used,
        merged_count=merged_count,
        best_score=best_score,
        region_area=region_area,
        prior_area=prior_area,
    )


def build_selection_mask_from_crop_alpha(
    crop_alpha: bytes,
    region: RegionSelectionInput,
    score: float,
    *,
    fallback_used: bool = False,
) -> SelectionMaskBuildResult:
    crop_width = region.crop_bbox[2]
    crop_height = region.crop_bbox[3]
    expected_size = crop_width * crop_height
    if len(crop_alpha) != expected_size:
        raise ValueError("selection alpha size does not match crop size")

    full = bytearray(region.canvas_width * region.canvas_height)
    crop_x, crop_y, _, _ = region.crop_bbox
    min_x = region.canvas_width
    min_y = region.canvas_height
    max_x = -1
    max_y = -1
    area = 0
    for y in range(crop_height):
        for x in range(crop_width):
            crop_index = y * crop_width + x
            alpha = crop_alpha[crop_index]
            if alpha == 0:
                continue
            if region.crop_region_mask[crop_index] == 0 or region.crop_alpha_mask[crop_index] == 0:
                continue
            canvas_x = crop_x + x
            canvas_y = crop_y + y
            full[canvas_y * region.canvas_width + canvas_x] = alpha
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
            "score": score,
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
    boundary_coverage: float
    positive_hit_rate: float
    negative_hit_rate: float


def _build_recall_prior(
    best: _RankedCandidate,
    candidates: Sequence[_RankedCandidate],
    region: RegionSelectionInput,
) -> tuple[list[bool], int]:
    merged = list(best.mask)
    merged_count = 1
    for candidate in sorted(candidates, key=lambda item: item.rank, reverse=True):
        if candidate.index == best.index:
            continue
        if not _candidate_supports_recall_union(candidate, best):
            continue
        added_area = _added_inside_area(merged, candidate.mask, region)
        base_area = _selected_inside_area(merged, region)
        if added_area < max(8, int(base_area * 0.01)):
            continue
        for index, selected in enumerate(candidate.mask):
            if selected:
                merged[index] = True
        merged_count += 1
    return merged, merged_count


def _candidate_supports_recall_union(candidate: _RankedCandidate, best: _RankedCandidate) -> bool:
    if candidate.stats.positive_hit_rate < MIN_RECALL_POSITIVE_HIT_RATE:
        return False
    if candidate.stats.negative_hit_rate > MAX_RECALL_NEGATIVE_HIT_RATE:
        return False
    if candidate.stats.region_coverage < max(
        MIN_REGION_COVERAGE, best.stats.region_coverage * 0.25
    ):
        return False
    return not (
        candidate.rank < best.rank - 0.35
        and candidate.stats.region_coverage <= best.stats.region_coverage
    )


def _selected_inside_area(candidate: Sequence[bool], region: RegionSelectionInput) -> int:
    return sum(
        1
        for index, selected in enumerate(candidate)
        if selected and region.crop_region_mask[index] > 0 and region.crop_alpha_mask[index] > 0
    )


def _added_inside_area(
    current: Sequence[bool], candidate: Sequence[bool], region: RegionSelectionInput
) -> int:
    return sum(
        1
        for index, selected in enumerate(candidate)
        if (
            selected
            and not current[index]
            and region.crop_region_mask[index] > 0
            and region.crop_alpha_mask[index] > 0
        )
    )


def _candidate_stats(candidate: Sequence[bool], region: RegionSelectionInput) -> _CandidateStats:
    region_mask = region.crop_region_mask
    alpha_mask = region.crop_alpha_mask
    width = region.crop_bbox[2]
    height = region.crop_bbox[3]
    mask_area = 0
    inside_area = 0
    region_area = _count_region(region_mask, alpha_mask)
    for index, selected in enumerate(candidate):
        if not selected or alpha_mask[index] == 0:
            continue
        mask_area += 1
        if region_mask[index] > 0:
            inside_area += 1
    boundary_indexes = _region_boundary_indexes(region_mask, alpha_mask, width, height)
    boundary_selected = sum(1 for index in boundary_indexes if candidate[index])
    positive_total = 0
    positive_hits = 0
    negative_total = 0
    negative_hits = 0
    for point, label in zip(region.point_coords, region.point_labels, strict=False):
        x = max(0, min(width - 1, int(round(point[0]))))
        y = max(0, min(height - 1, int(round(point[1]))))
        hit = candidate[y * width + x]
        if label == 1:
            positive_total += 1
            if hit:
                positive_hits += 1
        else:
            negative_total += 1
            if hit:
                negative_hits += 1
    return _CandidateStats(
        inside_area=inside_area,
        candidate_coverage=inside_area / mask_area if mask_area else 0.0,
        region_coverage=inside_area / region_area if region_area else 0.0,
        boundary_coverage=boundary_selected / len(boundary_indexes) if boundary_indexes else 0.0,
        positive_hit_rate=positive_hits / positive_total if positive_total else 1.0,
        negative_hit_rate=negative_hits / negative_total if negative_total else 0.0,
    )


def _count_region(region_mask: bytes, alpha_mask: bytes) -> int:
    return sum(1 for index, value in enumerate(region_mask) if value > 0 and alpha_mask[index] > 0)


def _region_boundary_indexes(
    region_mask: bytes, alpha_mask: bytes, width: int, height: int
) -> list[int]:
    indexes: list[int] = []
    for y in range(height):
        for x in range(width):
            index = y * width + x
            if region_mask[index] == 0 or alpha_mask[index] == 0:
                continue
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                indexes.append(index)
                continue
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                neighbor = ny * width + nx
                if region_mask[neighbor] == 0 or alpha_mask[neighbor] == 0:
                    indexes.append(index)
                    break
    return indexes


def _fill_small_holes(
    candidate: Sequence[bool], region_mask: bytes, alpha_mask: bytes, width: int, height: int
) -> list[bool]:
    output = list(candidate)
    region_area = _count_region(region_mask, alpha_mask)
    max_hole_area = min(
        MAX_HOLE_FILL_AREA, max(64, int(region_area * MAX_HOLE_FILL_REGION_FRACTION))
    )
    if max_hole_area <= 0:
        return output

    reachable = [False] * (width * height)
    queue: deque[int] = deque()
    for index in _region_boundary_indexes(region_mask, alpha_mask, width, height):
        if not output[index]:
            reachable[index] = True
            queue.append(index)

    while queue:
        index = queue.popleft()
        x = index % width
        y = index // width
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            neighbor = ny * width + nx
            if reachable[neighbor] or output[neighbor]:
                continue
            if region_mask[neighbor] == 0 or alpha_mask[neighbor] == 0:
                continue
            reachable[neighbor] = True
            queue.append(neighbor)

    visited = [False] * (width * height)
    for index in range(width * height):
        if (
            visited[index]
            or output[index]
            or reachable[index]
            or region_mask[index] == 0
            or alpha_mask[index] == 0
        ):
            continue
        component: list[int] = []
        queue.append(index)
        visited[index] = True
        while queue:
            current = queue.popleft()
            component.append(current)
            x = current % width
            y = current // width
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                neighbor = ny * width + nx
                if (
                    visited[neighbor]
                    or output[neighbor]
                    or reachable[neighbor]
                    or region_mask[neighbor] == 0
                    or alpha_mask[neighbor] == 0
                ):
                    continue
                visited[neighbor] = True
                queue.append(neighbor)
        if len(component) <= max_hole_area:
            for hole_index in component:
                output[hole_index] = True
    return output


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
