# Code Style

## Python

Python formatting and linting use Ruff.

Commands:

```bash
.venv/bin/python -m ruff format python worker tests
.venv/bin/python -m ruff check python worker tests
```

The default verification command runs Ruff in check mode:

```bash
PYTHON=.venv/bin/python ./scripts/test-code.sh
```

Rules:

- Keep worker modules typed and small.
- Prefer dataclasses for structured protocol/cache/runtime data.
- Use explicit exception classes for user-safe worker errors.
- Avoid broad dynamic imports outside runtime-dependent code paths.
- Do not pass image or mask bytes through JSON.

## C++

C++ formatting uses `.clang-format`.

Format touched C++ files before building:

```bash
clang-format -i plugin/SmartSelectTool.cpp plugin/SmartSelectTool.h
```

Rules:

- Do not mass-format imported Krita code or unrelated source files.
- Prefer typed structs over loose `QVariant`/JSON maps inside core logic.
- Use `enum class` for internal state.
- Preserve soft alpha bytes as `0..255`; do not binarize selection masks.
- Keep UI work on the Krita side and inference work in the worker process.

`clang-tidy` is not part of the required check set.

## Documentation

- Keep stable decisions in `docs/`.
- Update docs in the same change as architecture, protocol, runtime, logging,
  testing or user-facing behavior changes.
- Avoid documenting obsolete implementation history in release-facing docs.
