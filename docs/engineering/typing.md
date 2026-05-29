# Typing

## Principle

Статическая типизация желательна, но вводить её нужно как typed islands.

Цель: строго типизировать чистую core-логику и протокольные границы, не пытаясь
заставить mypy понимать все внутренности Krita/PyQt до того, как архитектура
станет чистой.

## Strict Areas

Строгая типизация обязательна для:

- worker protocol;
- model registry;
- runtime environment bootstrap;
- BiRefNet alpha inference and alpha-mask building;
- cache keys;
- snapshot metadata;
- pure Python core modules;
- pure C++ core data structures.

## Pragmatic Areas

Типизировать прагматично:

- Krita/PyQt adapter code;
- dynamic UI integration;
- ctypes native loader;
- dynamic imports for optional runtime dependencies such as torch, PIL,
  transformers and Hugging Face Hub;
- места, где API Krita не имеет внятных Python stubs.

## Python Policy

Python code проверяется `mypy`.

Current config:

- config file: `pyproject.toml`;
- `strict = true`;
- `mypy_path = "typings"`;
- checked paths:
  - `worker`;
  - `python/smart_select`.

Local Krita stubs живут в `typings/krita`.

Stub policy:

- Stub only APIs we touch.
- Не писать полный stub для Krita/PyQt заранее.
- При добавлении нового использования Krita API расширять локальный stub
  минимально и конкретно.

## C++ Policy

- Использовать конкретные типы вместо `QVariant`/loose JSON во внутренней
  логике.
- Использовать `enum class` для state.
- Selection alpha values должны сохраняться как raw `uint8` `0..255`; не
  binarize before apply.
- Если worker JSON становится сложнее, парсить его в typed DTO перед передачей
  глубже в tool/core logic.

## Commands

Первичная настройка:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

Проверка:

```bash
PYTHON=.venv/bin/python ./scripts/test-code.sh
```

Только Python formatting/linting:

```bash
.venv/bin/python -m ruff format --check python worker tests
.venv/bin/python -m ruff check python worker tests
```

Только mypy:

```bash
.venv/bin/python -m mypy
```

## Current Status

- Worker and Python loader are checked by mypy.
- Krita API is represented by minimal local stubs.
- C++ core has compile-time type checking and QtTest coverage.

Formatting and linting policy lives in [code-style.md](code-style.md).
