from __future__ import annotations

from dataclasses import dataclass

from observability import log_event

TRIMAP_CORE_ERODE_MIN = 1
TRIMAP_CORE_ERODE_MAX = 6
TRIMAP_SUPPORT_DILATE_MIN = 2
TRIMAP_SUPPORT_DILATE_MAX = 12
TRIMAP_CORE_ERODE_RATIO = 0.006
TRIMAP_SUPPORT_DILATE_RATIO = 0.012


@dataclass(frozen=True)
class TrimapBuildResult:
    trimap_mask: bytes
    core_mask: bytes
    unknown_mask: bytes
    support_mask: bytes
    core_erode_radius: int
    support_dilate_radius: int
    prior_area: int
    core_area: int
    unknown_area: int
    support_area: int


def build_trimap_from_prior(
    prior_mask: bytes,
    region_mask: bytes,
    alpha_mask: bytes,
    width: int,
    height: int,
) -> TrimapBuildResult:
    expected_size = width * height
    if width <= 0 or height <= 0:
        raise ValueError("trimap dimensions are invalid")
    if (
        len(prior_mask) != expected_size
        or len(region_mask) != expected_size
        or len(alpha_mask) != expected_size
    ):
        raise ValueError("trimap mask sizes do not match")

    prior = [value > 0 for value in prior_mask]
    region = [value > 0 for value in region_mask]
    document_alpha = [value > 0 for value in alpha_mask]
    prior = [
        selected and region[index] and document_alpha[index] for index, selected in enumerate(prior)
    ]

    base = max(width, height)
    core_radius = _clamp(
        round(base * TRIMAP_CORE_ERODE_RATIO),
        TRIMAP_CORE_ERODE_MIN,
        TRIMAP_CORE_ERODE_MAX,
    )
    support_radius = _clamp(
        round(base * TRIMAP_SUPPORT_DILATE_RATIO),
        TRIMAP_SUPPORT_DILATE_MIN,
        TRIMAP_SUPPORT_DILATE_MAX,
    )

    prior_area = _count_true(prior)
    core = _non_empty_erode(prior, width, height, core_radius)
    support = _dilate(prior, width, height, support_radius)
    support = [
        selected and region[index] and document_alpha[index]
        for index, selected in enumerate(support)
    ]
    core = [selected and support[index] for index, selected in enumerate(core)]
    unknown = [support[index] and not core[index] for index in range(expected_size)]

    core_area = _count_true(core)
    unknown_area = _count_true(unknown)
    support_area = _count_true(support)

    trimap = bytearray(expected_size)
    core_bytes = bytearray(expected_size)
    unknown_bytes = bytearray(expected_size)
    support_bytes = bytearray(expected_size)
    for index in range(expected_size):
        if core[index]:
            trimap[index] = 255
            core_bytes[index] = 255
        elif unknown[index]:
            trimap[index] = 128
            unknown_bytes[index] = 255
        if support[index]:
            support_bytes[index] = 255

    log_event(
        "trimap.result",
        component="worker",
        **{
            "selection.prior.area": prior_area,
            "trimap.core.area": core_area,
            "trimap.unknown.area": unknown_area,
            "trimap.support.area": support_area,
            "trimap.core_erode_radius": core_radius,
            "trimap.support_dilate_radius": support_radius,
        },
    )
    return TrimapBuildResult(
        trimap_mask=bytes(trimap),
        core_mask=bytes(core_bytes),
        unknown_mask=bytes(unknown_bytes),
        support_mask=bytes(support_bytes),
        core_erode_radius=core_radius,
        support_dilate_radius=support_radius,
        prior_area=prior_area,
        core_area=core_area,
        unknown_area=unknown_area,
        support_area=support_area,
    )


def _non_empty_erode(mask: list[bool], width: int, height: int, radius: int) -> list[bool]:
    for current_radius in range(radius, 0, -1):
        eroded = _erode(mask, width, height, current_radius)
        if any(eroded):
            return eroded
    return list(mask)


def _dilate(mask: list[bool], width: int, height: int, radius: int) -> list[bool]:
    if radius <= 0:
        return list(mask)
    horizontal = [False] * len(mask)
    for y in range(height):
        row = y * width
        count = sum(1 for x in range(0, min(width, radius + 1)) if mask[row + x])
        for x in range(width):
            if x > 0:
                remove_x = x - radius - 1
                if remove_x >= 0 and mask[row + remove_x]:
                    count -= 1
                add_x = x + radius
                if add_x < width and mask[row + add_x]:
                    count += 1
            horizontal[row + x] = count > 0

    output = [False] * len(mask)
    for x in range(width):
        count = sum(1 for y in range(0, min(height, radius + 1)) if horizontal[y * width + x])
        for y in range(height):
            if y > 0:
                remove_y = y - radius - 1
                if remove_y >= 0 and horizontal[remove_y * width + x]:
                    count -= 1
                add_y = y + radius
                if add_y < height and horizontal[add_y * width + x]:
                    count += 1
            output[y * width + x] = count > 0
    return output


def _erode(mask: list[bool], width: int, height: int, radius: int) -> list[bool]:
    if radius <= 0:
        return list(mask)
    kernel = radius * 2 + 1
    horizontal = [False] * len(mask)
    for y in range(height):
        row = y * width
        count = sum(1 for x in range(0, min(width, radius + 1)) if mask[row + x])
        for x in range(width):
            if x > 0:
                remove_x = x - radius - 1
                if remove_x >= 0 and mask[row + remove_x]:
                    count -= 1
                add_x = x + radius
                if add_x < width and mask[row + add_x]:
                    count += 1
            horizontal[row + x] = x >= radius and x < width - radius and count == kernel

    output = [False] * len(mask)
    for x in range(width):
        count = sum(1 for y in range(0, min(height, radius + 1)) if horizontal[y * width + x])
        for y in range(height):
            if y > 0:
                remove_y = y - radius - 1
                if remove_y >= 0 and horizontal[remove_y * width + x]:
                    count -= 1
                add_y = y + radius
                if add_y < height and horizontal[add_y * width + x]:
                    count += 1
            output[y * width + x] = y >= radius and y < height - radius and count == kernel
    return output


def _count_true(values: list[bool]) -> int:
    return sum(1 for value in values if value)


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))
