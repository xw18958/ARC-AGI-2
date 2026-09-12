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
    matches = ANSWER_RE.findall(text)
    if matches:
        # Once the model uses the explicit contract, never recover a different
        # grid from its reasoning when the answer block itself is malformed.
        candidates = [array for answer in matches for array in _balanced_arrays(answer)]
    else:
        lowered = text.lower()
        if "<answer" in lowered:
            return None
        if "<think" in lowered:
            # A completion truncated inside native thinking has no final answer.
            # Mining demonstration/query grids from that reasoning would create
            # spurious progress reward for a trajectory excluded from the loss.
            if "</think>" not in lowered:
                return None
            text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
        candidates = _balanced_arrays(text)

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
