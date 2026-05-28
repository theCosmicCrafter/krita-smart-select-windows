from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelFamily = Literal["mock", "sam2.1"]
ModelProfile = Literal["mock", "quality"]

DEFAULT_MODEL_ID = "sam2.1_hiera_large"
SAM21_LARGE_CHECKPOINT_URL = "https://huggingface.co/facebook/sam2.1-hiera-large/resolve/38e0b24f84dfe8a95d5b6cb53bc1b772cbe15dc2/sam2.1_hiera_large.pt?download=true"
SAM21_LARGE_EXPECTED_BYTES = 898_089_053
SAM21_LARGE_ETAG = '"5a2b7e292d0cbd3cce5f410502ced3eb9b92031a7b21fcea62cb97eed73de1d3"'


class ModelRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    title: str
    family: ModelFamily
    checkpoint_url: str | None
    checkpoint_filename: str | None
    config_name: str | None
    expected_bytes: int | None
    etag: str | None
    recommended_ram_gb: int | None
    recommended_vram_gb: int | None
    profile: ModelProfile
    requires_download: bool


_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id=DEFAULT_MODEL_ID,
        title="SAM2.1 Hiera Large",
        family="sam2.1",
        checkpoint_url=SAM21_LARGE_CHECKPOINT_URL,
        checkpoint_filename="sam2.1_hiera_large.pt",
        config_name="configs/sam2.1/sam2.1_hiera_l.yaml",
        expected_bytes=SAM21_LARGE_EXPECTED_BYTES,
        etag=SAM21_LARGE_ETAG,
        recommended_ram_gb=32,
        recommended_vram_gb=12,
        profile="quality",
        requires_download=True,
    ),
    ModelSpec(
        id="mock",
        title="Mock Segmentation",
        family="mock",
        checkpoint_url=None,
        checkpoint_filename=None,
        config_name=None,
        expected_bytes=None,
        etag=None,
        recommended_ram_gb=None,
        recommended_vram_gb=None,
        profile="mock",
        requires_download=False,
    ),
)

_MODELS_BY_ID = {model.id: model for model in _MODELS}


def default_model_id() -> str:
    return DEFAULT_MODEL_ID


def all_model_specs() -> tuple[ModelSpec, ...]:
    return _MODELS


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return _MODELS_BY_ID[model_id]
    except KeyError as exc:
        raise ModelRegistryError("unknown model") from exc
