# Typing

## Principle

Static typing is valuable, but it should be introduced as typed islands.

The goal is to type pure core logic and protocol boundaries strictly without
forcing mypy to understand every Krita/PyQt integration detail.

## Strict Areas

Strict typing is required for:

- worker protocol;
- model registry;
- runtime environment bootstrap;
- BiRefNet alpha inference and alpha-mask building;
- cache keys;
- snapshot metadata;
- pure Python core modules;
- pure C++ core data structures.

## Pragmatic Areas

Use pragmatic typing for:

- Krita/PyQt adapter code;
- dynamic UI integration;
- ctypes native loader;
- dynamic imports for optional runtime dependencies such as torch, PIL,
  transformers and Hugging Face Hub;
- places where the Krita API does not have clear Python stubs.

## Python Policy

Python code is checked with `mypy`.

Current config:

- config file: `pyproject.toml`;
- `strict = true`;
- `mypy_path = "typings"`;
- checked paths:
  - `worker`;
  - `python/smart_select`.

Local Krita stubs live in `typings/krita`.

Stub policy:

- Stub only APIs we touch.
- Do not write a full Krita/PyQt stub up front.
- When adding new Krita API usage, extend the local stub narrowly and
  specifically.

## C++ Policy

- Use concrete types instead of `QVariant`/loose JSON in internal logic.
- Use `enum class` for state.
- Selection alpha values must stay as raw `uint8` `0..255`; do not binarize
  before apply.
- If worker JSON becomes more complex, parse it into typed DTOs before passing
  it deeper into tool/core logic.

## Commands

Initial setup:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

Check:

```bash
PYTHON=.venv/bin/python ./scripts/test-code.sh
```

Python formatting/linting only:

```bash
.venv/bin/python -m ruff format --check python worker tests
.venv/bin/python -m ruff check python worker tests
```

mypy only:

```bash
.venv/bin/python -m mypy
```

## Current Status

- Worker and Python loader are checked by mypy.
- Krita API is represented by minimal local stubs.
- C++ core has compile-time type checking and QtTest coverage.

Formatting and linting policy lives in [code-style.md](code-style.md).
