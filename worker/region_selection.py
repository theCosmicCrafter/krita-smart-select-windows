from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

AlphaReader = Callable[[str | Path], tuple[int, int, bytes]]


class RegionSelectionError(ValueError):
    pass


@dataclass(frozen=True)
class RegionSelectionInput:
    crop_image_array: object
    crop_alpha_mask: bytes
    crop_region_mask: bytes
    crop_bbox: tuple[int, int, int, int]
    region_bbox: tuple[int, int, int, int]
    prompt_box_xyxy: tuple[float, float, float, float]
    point_coords: tuple[tuple[float, float], ...]
    point_labels: tuple[int, ...]
    canvas_width: int
    canvas_height: int


def load_region_selection_input(
    image_path: str | Path,
    alpha_mask_path: str | Path,
    region_mask_path: str | Path,
    canvas_width: int,
    canvas_height: int,
    requested_region_bbox: Sequence[object],
    alpha_reader: AlphaReader,
) -> RegionSelectionInput:
    image_module = importlib.import_module("PIL.Image")
    numpy_module = importlib.import_module("numpy")

    image = image_module.open(image_path).convert("RGB")
    if image.size != (canvas_width, canvas_height):
        raise RegionSelectionError("Snapshot image size does not match canvas size")

    alpha_width, alpha_height, alpha_mask = alpha_reader(alpha_mask_path)
    if alpha_width != canvas_width or alpha_height != canvas_height:
        raise RegionSelectionError("Alpha mask size does not match canvas size")

    region_width, region_height, region_mask = alpha_reader(region_mask_path)
    if region_width != canvas_width or region_height != canvas_height:
        raise RegionSelectionError("Region mask size does not match canvas size")

    computed_bbox = _mask_bbox(region_mask, canvas_width, canvas_height)
    if computed_bbox is None:
        raise RegionSelectionError("lasso region is empty")

    requested_bbox = _parse_bbox(requested_region_bbox, canvas_width, canvas_height)
    region_bbox = _intersect_bboxes(computed_bbox, requested_bbox) or computed_bbox
    if region_bbox[2] <= 0 or region_bbox[3] <= 0:
        raise RegionSelectionError("lasso region is empty")

    crop_bbox = _padded_bbox(region_bbox, canvas_width, canvas_height)
    crop_x, crop_y, crop_w, crop_h = crop_bbox
    crop_image = image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
    crop_region = _crop_bytes(region_mask, canvas_width, crop_bbox)
    crop_alpha = _crop_bytes(alpha_mask, canvas_width, crop_bbox)

    local_region_bbox = (
        region_bbox[0] - crop_x,
        region_bbox[1] - crop_y,
        region_bbox[2],
        region_bbox[3],
    )
    point_coords, point_labels = _prompt_points(crop_region, crop_alpha, crop_w, crop_h)
    if not point_coords:
        raise RegionSelectionError("lasso region is empty")

    box_x, box_y, box_w, box_h = local_region_bbox
    prompt_box = (
        float(box_x),
        float(box_y),
        float(box_x + max(1, box_w)),
        float(box_y + max(1, box_h)),
    )

    return RegionSelectionInput(
        crop_image_array=numpy_module.array(crop_image, copy=True),
        crop_alpha_mask=crop_alpha,
        crop_region_mask=crop_region,
        crop_bbox=crop_bbox,
        region_bbox=region_bbox,
        prompt_box_xyxy=prompt_box,
        point_coords=tuple(point_coords),
        point_labels=tuple(point_labels),
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )


def _parse_bbox(
    value: Sequence[object], canvas_width: int, canvas_height: int
) -> tuple[int, int, int, int]:
    if len(value) != 4:
        raise RegionSelectionError("Region bbox is invalid")
    try:
        x = _int_from_object(value[0])
        y = _int_from_object(value[1])
        width = _int_from_object(value[2])
        height = _int_from_object(value[3])
    except (TypeError, ValueError) as exc:
        raise RegionSelectionError("Region bbox is invalid") from exc

    x0 = max(0, min(canvas_width, x))
    y0 = max(0, min(canvas_height, y))
    x1 = max(x0, min(canvas_width, x + max(0, width)))
    y1 = max(y0, min(canvas_height, y + max(0, height)))
    return x0, y0, x1 - x0, y1 - y0


def _int_from_object(value: object) -> int:
    if isinstance(value, int | float | str | bytes | bytearray):
        return int(value)
    raise TypeError("value is not integer-like")


def _mask_bbox(mask: bytes, width: int, height: int) -> tuple[int, int, int, int] | None:
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    for y in range(height):
        row_offset = y * width
        for x in range(width):
            if mask[row_offset + x] == 0:
                continue
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    if max_x < min_x or max_y < min_y:
        return None
    return min_x, min_y, max_x - min_x + 1, max_y - min_y + 1


def _intersect_bboxes(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[0] + first[2], second[0] + second[2])
    y1 = min(first[1] + first[3], second[1] + second[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def _padded_bbox(
    bbox: tuple[int, int, int, int], canvas_width: int, canvas_height: int
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    padding = max(32, round(max(width, height) * 0.10))
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(canvas_width, x + width + padding)
    y1 = min(canvas_height, y + height + padding)
    return x0, y0, x1 - x0, y1 - y0


def _crop_bytes(mask: bytes, source_width: int, bbox: tuple[int, int, int, int]) -> bytes:
    x, y, width, height = bbox
    output = bytearray(width * height)
    for row in range(height):
        source_offset = (y + row) * source_width + x
        target_offset = row * width
        output[target_offset : target_offset + width] = mask[source_offset : source_offset + width]
    return bytes(output)


def _prompt_points(
    region_mask: bytes, alpha_mask: bytes, width: int, height: int
) -> tuple[list[tuple[float, float]], list[int]]:
    positive = _positive_points(region_mask, alpha_mask, width, height)
    negative = _negative_points(region_mask, alpha_mask, width, height)
    return positive + negative, [1] * len(positive) + [0] * len(negative)


def _positive_points(
    region_mask: bytes, alpha_mask: bytes, width: int, height: int
) -> list[tuple[float, float]]:
    selected: list[tuple[int, int]] = []
    sum_x = 0
    sum_y = 0
    for y in range(height):
        row_offset = y * width
        for x in range(width):
            index = row_offset + x
            if region_mask[index] == 0 or alpha_mask[index] == 0:
                continue
            selected.append((x, y))
            sum_x += x
            sum_y += y
    if not selected:
        return []

    points: list[tuple[float, float]] = [
        (float(round(sum_x / len(selected))), float(round(sum_y / len(selected))))
    ]
    for fraction_x, fraction_y in (
        (0.25, 0.25),
        (0.50, 0.25),
        (0.75, 0.25),
        (0.25, 0.50),
        (0.75, 0.50),
        (0.25, 0.75),
        (0.50, 0.75),
        (0.75, 0.75),
    ):
        candidate = _nearest_selected_point(selected, width, height, fraction_x, fraction_y)
        if candidate is not None:
            point = (float(candidate[0]), float(candidate[1]))
            if point not in points:
                points.append(point)
    return points[:9]


def _nearest_selected_point(
    selected: Sequence[tuple[int, int]],
    width: int,
    height: int,
    fraction_x: float,
    fraction_y: float,
) -> tuple[int, int] | None:
    target_x = width * fraction_x
    target_y = height * fraction_y
    if not selected:
        return None
    return min(selected, key=lambda point: (point[0] - target_x) ** 2 + (point[1] - target_y) ** 2)


def _negative_points(
    region_mask: bytes, alpha_mask: bytes, width: int, height: int
) -> list[tuple[float, float]]:
    bbox = _mask_bbox(region_mask, width, height)
    if bbox is None:
        return []
    x, y, bbox_w, bbox_h = bbox
    x0 = x
    y0 = y
    x1 = x + bbox_w - 1
    y1 = y + bbox_h - 1
    inner_candidates = (
        (x0, y0),
        ((x0 + x1) // 2, y0),
        (x1, y0),
        (x0, (y0 + y1) // 2),
        (x1, (y0 + y1) // 2),
        (x0, y1),
        ((x0 + x1) // 2, y1),
        (x1, y1),
    )
    outer_candidates = (
        (x0 - 8, y0 - 8),
        ((x0 + x1) // 2, y0 - 8),
        (x1 + 8, y0 - 8),
        (x0 - 8, (y0 + y1) // 2),
        (x1 + 8, (y0 + y1) // 2),
        (x0 - 8, y1 + 8),
        ((x0 + x1) // 2, y1 + 8),
        (x1 + 8, y1 + 8),
    )
    points: list[tuple[float, float]] = []
    selected = _selected_points(region_mask, alpha_mask, width, height)
    for candidate_x, candidate_y in inner_candidates:
        candidate = _nearest_selected_point(
            selected, width, height, candidate_x / max(1, width), candidate_y / max(1, height)
        )
        if candidate is None:
            continue
        point = (float(candidate[0]), float(candidate[1]))
        if point not in points:
            points.append(point)
    for candidate_x, candidate_y in outer_candidates:
        px = max(0, min(width - 1, candidate_x))
        py = max(0, min(height - 1, candidate_y))
        index = py * width + px
        if region_mask[index] != 0 or alpha_mask[index] == 0:
            continue
        point = (float(px), float(py))
        if point not in points:
            points.append(point)
    return points[:12]


def _selected_points(
    region_mask: bytes, alpha_mask: bytes, width: int, height: int
) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for y in range(height):
        row_offset = y * width
        for x in range(width):
            index = row_offset + x
            if region_mask[index] > 0 and alpha_mask[index] > 0:
                points.append((x, y))
    return points
