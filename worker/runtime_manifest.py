from __future__ import annotations

RUNTIME_VERSION = "birefnet-runtime-v1"
RUNTIME_PYTHON_VERSION = "3.12"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"

TORCH_PACKAGES = ("torch>=2.5.1,<3", "torchvision>=0.20.1,<1")
PYTHON_PACKAGES = (
    "numpy>=1.26,<3",
    "pillow>=10,<13",
    "transformers>=4.46,<5",
    "huggingface-hub>=0.26,<1",
    "safetensors>=0.4,<1",
    "timm>=1.0,<2",
    "kornia>=0.7,<1",
    "einops>=0.7,<1",
    "scipy>=1.11,<2",
    "scikit-image>=0.22,<1",
    "accelerate>=1,<2",
)
