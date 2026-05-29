# Contributing

Thanks for helping improve Krita Smart Select.

## Development Flow

1. Read [README.md](README.md) and [docs/architecture.md](docs/architecture.md).
2. Keep changes scoped to the current behavior unless a product decision changes.
3. Update documentation in the same change when architecture, protocol,
   runtime, testing, typing, logging or user-facing behavior changes.
4. Run the default checks before submitting:

```bash
PYTHON=.venv/bin/python ./scripts/test-code.sh
```

5. For native plugin changes, also build the plugin:

```bash
BUILD_JOBS=20 ./scripts/build-krita-5.3.1.sh
```

## Code Expectations

- Follow [docs/engineering/code-style.md](docs/engineering/code-style.md).
- Keep worker protocol payloads small; pass large data through temporary files.
- Preserve soft alpha values in selection masks.
- Keep runtime installs and model checkpoints out of the source tree.
- Use structured logs with `event.name` and dotted field names.
- Do not log image contents, pixel arrays, mask bytes, secrets or environment
  variables.

## Tests

Default tests must not download the real model or install runtime dependencies.
Use fake modules and fixtures for worker unit tests.

The real BiRefNet smoke test is opt-in:

```bash
SMART_SELECT_RUN_BIREFNET_SMOKE=1 ./scripts/test-birefnet-smoke.sh
```

## Commit Messages

Use Conventional Commits, for example:

```text
feat: add explicit ai selection action
fix: release birefnet cuda memory after job
docs: refresh release documentation
```
