from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RefinerFamily = Literal["mam"]
RefinerProfile = Literal["quality"]

DEFAULT_REFINER_ID = "mam_sam_vith"
MATTING_ANYTHING_SOURCE_REF = "eb01e1c1f0016e2a4d00615b115f3b5ddcd031e4"


class RefinerRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class RefinerSpec:
    id: str
    title: str
    family: RefinerFamily
    profile: RefinerProfile
    source_repo_url: str
    source_ref: str
    config_name: str
    sam_model_type: str
    sam_checkpoint_url: str
    sam_checkpoint_filename: str
    sam_expected_bytes: int
    sam_etag: str | None
    mam_checkpoint_url: str
    mam_checkpoint_filename: str
    mam_expected_bytes: int
    mam_etag: str | None
    recommended_ram_gb: int
    recommended_vram_gb: int
    requires_download: bool


_REFINERS: tuple[RefinerSpec, ...] = (
    RefinerSpec(
        id="mam_sam_vith",
        title="MAM SAM ViT-H",
        family="mam",
        profile="quality",
        source_repo_url="https://github.com/SHI-Labs/Matting-Anything.git",
        source_ref=MATTING_ANYTHING_SOURCE_REF,
        config_name="config/MAM-ViTH-8gpu.toml",
        sam_model_type="sam_vit_h",
        sam_checkpoint_url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        sam_checkpoint_filename="sam_vit_h_4b8939.pth",
        sam_expected_bytes=2_564_550_879,
        sam_etag='"11bf577cf1da9493a48d7a571ba83cd5-306"',
        mam_checkpoint_url=(
            "https://drive.usercontent.google.com/download"
            "?id=1fmBLZMWfGE2QOsc3pPqhzzlvBYojLBox&export=download&confirm=t"
        ),
        mam_checkpoint_filename="mam_vith.pth",
        mam_expected_bytes=32_785_332,
        mam_etag=None,
        recommended_ram_gb=32,
        recommended_vram_gb=16,
        requires_download=True,
    ),
)


def default_refiner_id() -> str:
    return DEFAULT_REFINER_ID


def all_refiner_specs() -> tuple[RefinerSpec, ...]:
    return _REFINERS


def get_refiner_spec(refiner_id: str) -> RefinerSpec:
    for spec in _REFINERS:
        if spec.id == refiner_id:
            return spec
    raise RefinerRegistryError(f"Unknown refiner: {refiner_id}")
