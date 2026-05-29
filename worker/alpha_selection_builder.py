from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import NotRequired, TypedDict

from observability import log_event
from region_selection import RegionSelectionInput

MIN_DETACHED_COMPONENT_AREA = 32


class EmptySelectionError(ValueError):
    pass


class SegmentMetadata(TypedDict):
    id: int
    bbox: list[int]
    area: int
    score: NotRequired[float]


@dataclass(frozen=True)
class AlphaSelectionBuildResult:
    selection_mask: bytes
    metadata: list[SegmentMetadata]
    raw_nonzero_pixels: int
    final_nonzero_pixels: int
    soft_pixels: int
    alpha_sum: int
    max_alpha: int
    mean_nonzero_alpha: float


def build_selection_mask_from_crop_alpha(
    crop_alpha: bytes,
    region: RegionSelectionInput,
    score: float = 1.0,
) -> AlphaSelectionBuildResult:
    crop_width = region.crop_bbox[2]
    crop_height = region.crop_bbox[3]
    expected_size = crop_width * crop_height
    if len(crop_alpha) != expected_size:
        raise ValueError("selection alpha size does not match crop size")

    clipped = bytearray(expected_size)
    raw_nonzero = 0
    for index, alpha in enumerate(crop_alpha):
        if alpha > 0:
            raw_nonzero += 1
        if alpha == 0:
            continue
        if region.crop_region_mask[index] == 0 or region.crop_alpha_mask[index] == 0:
            continue
        clipped[index] = alpha

    clipped = _remove_tiny_components(clipped, crop_width, crop_height)
    full = bytearray(region.canvas_width * region.canvas_height)
    crop_x, crop_y, _, _ = region.crop_bbox
    min_x = region.canvas_width
    min_y = region.canvas_height
    max_x = -1
    max_y = -1
    final_nonzero = 0
    soft_pixels = 0
    alpha_sum = 0
    max_alpha = 0

    for y in range(crop_height):
        for x in range(crop_width):
            crop_index = y * crop_width + x
            alpha = clipped[crop_index]
            if alpha == 0:
                continue
            canvas_x = crop_x + x
            canvas_y = crop_y + y
            full[canvas_y * region.canvas_width + canvas_x] = alpha
            min_x = min(min_x, canvas_x)
            min_y = min(min_y, canvas_y)
            max_x = max(max_x, canvas_x)
            max_y = max(max_y, canvas_y)
            final_nonzero += 1
            alpha_sum += alpha
            max_alpha = max(max_alpha, alpha)
            if alpha < 255:
                soft_pixels += 1

    if final_nonzero <= 0:
        raise EmptySelectionError("selection is empty")

    metadata: list[SegmentMetadata] = [
        {
            "id": 1,
            "bbox": [min_x, min_y, max_x - min_x + 1, max_y - min_y + 1],
            "area": final_nonzero,
            "score": score,
        }
    ]
    mean_nonzero_alpha = alpha_sum / final_nonzero
    log_event(
        "alpha_mask.result",
        component="worker",
        **{
            "selection.area": final_nonzero,
            "selection.soft_pixels": soft_pixels,
            "selection.alpha_sum": alpha_sum,
            "selection.max_alpha": max_alpha,
        },
    )
    return AlphaSelectionBuildResult(
        selection_mask=bytes(full),
        metadata=metadata,
        raw_nonzero_pixels=raw_nonzero,
        final_nonzero_pixels=final_nonzero,
        soft_pixels=soft_pixels,
        alpha_sum=alpha_sum,
        max_alpha=max_alpha,
        mean_nonzero_alpha=mean_nonzero_alpha,
    )


def _remove_tiny_components(mask: bytearray, width: int, height: int) -> bytearray:
    if width * height < MIN_DETACHED_COMPONENT_AREA * 4:
        return mask
    visited = [False] * len(mask)
    output = bytearray(mask)
    for start, alpha in enumerate(mask):
        if alpha == 0 or visited[start]:
            continue
        component = _collect_component(mask, visited, width, height, start)
        if len(component) >= MIN_DETACHED_COMPONENT_AREA:
            continue
        for index in component:
            output[index] = 0
    return output


def _collect_component(
    mask: bytearray, visited: list[bool], width: int, height: int, start: int
) -> list[int]:
    queue: deque[int] = deque([start])
    visited[start] = True
    component: list[int] = []
    while queue:
        index = queue.popleft()
        component.append(index)
        x = index % width
        y = index // width
        for nx, ny in _neighbors4(width, height, x, y):
            next_index = ny * width + nx
            if visited[next_index] or mask[next_index] == 0:
                continue
            visited[next_index] = True
            queue.append(next_index)
    return component


def _neighbors4(width: int, height: int, x: int, y: int) -> tuple[tuple[int, int], ...]:
    neighbors: list[tuple[int, int]] = []
    if x > 0:
        neighbors.append((x - 1, y))
    if x + 1 < width:
        neighbors.append((x + 1, y))
    if y > 0:
        neighbors.append((x, y - 1))
    if y + 1 < height:
        neighbors.append((x, y + 1))
    return tuple(neighbors)
