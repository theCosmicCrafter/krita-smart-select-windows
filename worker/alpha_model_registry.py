from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelFamily = Literal["birefnet"]
ModelProfile = Literal["quality"]

DEFAULT_MODEL_ID = "birefnet_hr_matting"


class ModelRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    title: str
    family: ModelFamily
    repo_id: str
    revision: str
    checkpoint_filename: str
    checkpoint_expected_bytes: int
    checkpoint_etag: str | None
    config_filename: str
    config_expected_bytes: int
    license: str
    recommended_ram_gb: int
    recommended_vram_gb: int
    input_resolution: int
    profile: ModelProfile
    requires_download: bool
    trust_remote_code: bool


_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id=DEFAULT_MODEL_ID,
        title="BiRefNet HR Matting",
        family="birefnet",
        repo_id="ZhengPeng7/BiRefNet_HR-matting",
        revision="5d6b6f8adcb5b417c871b1d84ceaae9871355b7f",
        checkpoint_filename="model.safetensors",
        checkpoint_expected_bytes=444_473_596,
        checkpoint_etag='"99cba4ff97616fa0e1e9b9480da68bd298ae62b4ebdc90ee3a829a939b2a7c72"',
        config_filename="config.json",
        config_expected_bytes=416,
        license="mit",
        recommended_ram_gb=16,
        recommended_vram_gb=8,
        input_resolution=2048,
        profile="quality",
        requires_download=True,
        trust_remote_code=True,
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
