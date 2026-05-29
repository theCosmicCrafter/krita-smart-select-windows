# References And Licensing

## Active Model

BiRefNet HR Matting:

- repo: `ZhengPeng7/BiRefNet_HR-matting`
- revision: `5d6b6f8adcb5b417c871b1d84ceaae9871355b7f`
- license: MIT
- used through Hugging Face `AutoModelForImageSegmentation` with
  `trust_remote_code=True` for this pinned curated repo only.

## Reference Learnings

Earlier research into SAM2, MAM and InvokeAI informed the pivot away from
prompt/trimap tuning toward a direct alpha model. Those paths are not part of
the current MVP implementation.

## Before Public Release

- Reconfirm BiRefNet model license and redistribution constraints.
- Consider vendoring/auditing pinned HF custom code.
- Document manual cleanup for unused old SAM2/MAM caches.
