from __future__ import annotations

RUNTIME_VERSION = "sam2-runtime-v3"
RUNTIME_PYTHON_VERSION = "3.12"
SAM2_SOURCE_URL = "https://github.com/facebookresearch/sam2.git"
SAM2_SOURCE_REF = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
MAM_SOURCE_URL = "https://github.com/SHI-Labs/Matting-Anything.git"
MAM_SOURCE_REF = "eb01e1c1f0016e2a4d00615b115f3b5ddcd031e4"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
SAM2_BUILD_CUDA = "1"
SAM2_BUILD_ALLOW_ERRORS = "0"
SAM2_REQUIRE_CUDA_EXTENSION = True

TORCH_PACKAGES = ("torch>=2.5.1", "torchvision>=0.20.1")
PYTHON_PACKAGES = ("numpy>=1.26,<3", "pillow>=10,<13")
SAM2_PACKAGE = f"git+{SAM2_SOURCE_URL}@{SAM2_SOURCE_REF}"
