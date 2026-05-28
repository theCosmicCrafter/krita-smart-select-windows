from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_INFERENCE_LONG_SIDE = 2048

AlphaReader = Callable[[str | Path], tuple[int, int, bytes]]


class ImagePreprocessingError(ValueError):
    pass


@dataclass(frozen=True)
class PreprocessedImage:
    image_array: Any
    alpha_mask: bytes
    canvas_width: int
    canvas_height: int
    inference_width: int
    inference_height: int
    resized: bool


def load_preprocessed_image(
    image_path: str | Path,
    alpha_mask_path: str | Path,
    canvas_width: int,
    canvas_height: int,
    alpha_reader: AlphaReader,
) -> PreprocessedImage:
    image_module = importlib.import_module("PIL.Image")
    numpy_module = importlib.import_module("numpy")

    image = image_module.open(image_path).convert("RGB")
    if image.size != (canvas_width, canvas_height):
        raise ImagePreprocessingError("Snapshot image size does not match canvas size")

    alpha_width, alpha_height, alpha_mask = alpha_reader(alpha_mask_path)
    if alpha_width != canvas_width or alpha_height != canvas_height:
        raise ImagePreprocessingError("Alpha mask size does not match canvas size")

    inference_width, inference_height = inference_size(canvas_width, canvas_height)
    resized = (inference_width, inference_height) != (canvas_width, canvas_height)
    if resized:
        resampling = getattr(image_module, "Resampling", image_module)
        image = image.resize((inference_width, inference_height), resampling.LANCZOS)
        alpha_image = image_module.frombytes("L", (canvas_width, canvas_height), alpha_mask)
        alpha_image = alpha_image.resize((inference_width, inference_height), resampling.NEAREST)
        alpha_mask = bytes(alpha_image.tobytes())

    return PreprocessedImage(
        image_array=numpy_module.asarray(image),
        alpha_mask=alpha_mask,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        inference_width=inference_width,
        inference_height=inference_height,
        resized=resized,
    )


def inference_size(width: int, height: int) -> tuple[int, int]:
    long_side = max(width, height)
    if long_side <= MAX_INFERENCE_LONG_SIDE:
        return width, height

    scale = MAX_INFERENCE_LONG_SIDE / long_side
    return max(1, round(width * scale)), max(1, round(height * scale))
