from __future__ import annotations

import json
import re
import sys
from typing import Any, Literal, cast

Severity = Literal["debug", "info", "warning", "error"]

_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_./:@+-]+$")


def log_event(
    event_name: str,
    *,
    severity: object = "info",
    component: object = "worker",
    **fields: Any,
) -> None:
    severity_text: Severity = (
        cast(Severity, severity) if severity in {"debug", "info", "warning", "error"} else "info"
    )
    parts = [
        "[SmartSelectWorker]",
        f"event.name={_format_value(event_name)}",
        "event.domain=smart_select",
        f"severity={severity_text}",
        f"component={_format_value(component)}",
    ]
    parts.extend(f"{key}={_format_value(value)}" for key, value in fields.items())
    print(" ".join(parts), file=sys.stderr, flush=True)


def _format_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        if value and _SAFE_VALUE.fullmatch(value):
            return value
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, tuple):
        return json.dumps(list(value), ensure_ascii=False, default=str)
    return json.dumps(value, ensure_ascii=False, default=str)
