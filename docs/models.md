# Models

## Current Product Model

Smart Select currently uses one plug-and-play foreground alpha model:

- local id: `birefnet_hr_matting`
- title: BiRefNet HR Matting
- role: direct alpha extraction from a lasso crop
- algorithm version: `birefnet-hr-matting-lasso-v1`

SAM2, MAM, trimap refinement, BEN2, RMBG and SAM3 are research follow-ups, not
active implementation paths.

## BiRefNet HR Matting

Pinned metadata:

- Hugging Face repo: `ZhengPeng7/BiRefNet_HR-matting`
- revision: `5d6b6f8adcb5b417c871b1d84ceaae9871355b7f`
- license: MIT
- `model.safetensors`: `444473596` bytes
- `model.safetensors` ETag:
  `"99cba4ff97616fa0e1e9b9480da68bd298ae62b4ebdc90ee3a829a939b2a7c72"`
- `config.json`: `416` bytes
- input resolution: `2048x2048`
- recommended RAM: 16 GB
- recommended VRAM: 8 GB

The model card says this variant is trained at `2048x2048` for higher
resolution image matting with transparency.

## Cache

Worker downloads the pinned Hugging Face snapshot into:

```text
<model_cache_dir>/birefnet_hr_matting/snapshot/
```

Allowed files:

- `*.py`
- `*.json`
- `*.safetensors`
- `README.md`

The cache is marked verified only after exact file-size checks pass for
`model.safetensors` and `config.json`.

## Security Rule

`trust_remote_code=True` is allowed only for the pinned curated BiRefNet repo
and revision above. Do not expose arbitrary user-provided model repos in the MVP
path.

## Follow-Ups

Potential future research:

- model picker;
- BEN2;
- RMBG;
- SAM3;
- vendor/audit BiRefNet custom code instead of dynamic HF custom code;
- persistent model cache management UI.
