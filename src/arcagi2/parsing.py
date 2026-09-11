from __future__ import annotations

import json
import re
from typing import Any

from .types import Grid

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def _is_valid_grid(value: Any) -> bool:
    if not isinstance(value, list) or not value or not all(isinstance(r, list) for r in value):
        return False
    width = len(value[0])
    if width == 0 or len(value) > 30 or width > 30 or any(len(r) != width for r in value):
        return False
    return all(type(v) is int and 0 <= v <= 9 for r in value for v in r)


def _balanced_arrays(text: str) -> list[str]:
    out: list[str] = []
    start = None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                out.append(text[start : i + 1])
                start = None
    return out


def parse_grid(text: str) -> Grid | None:
    """Extract a strict ARC grid, preferring the explicit <answer> block."""
    candidates: list[str] = []
    matches = ANSWER_RE.findall(text)
    for answer in matches:
        candidates.extend(_balanced_arrays(answer))
    if not candidates:
        # Qwen's native thinking block may be followed by a bare final grid.
        tail = text.rsplit("</think>", 1)[-1]
        candidates.extend(_balanced_arrays(tail))
    if not candidates:
        candidates.extend(_balanced_arrays(text))

    # Prefer the last valid array: reasoning may quote demonstration grids first.
    for candidate in reversed(candidates):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if _is_valid_grid(value):
            return value
    return None


def canonical_grid(grid: Grid) -> str:
    return json.dumps(grid, separators=(",", ":"))
