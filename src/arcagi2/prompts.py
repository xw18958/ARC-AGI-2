from __future__ import annotations

import json
from importlib.resources import files

from .types import Grid, Pair

_PROMPT_ASSETS = {
    "v1": "student_universal_method.md",
    "v2": "student_universal_method_v2.md",
}


def load_reasoning_method(prompt_method: str = "v1") -> str:
    try:
        asset = _PROMPT_ASSETS[prompt_method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown prompt method {prompt_method!r}; choose one of {sorted(_PROMPT_ASSETS)}"
        ) from exc
    return files("arcagi2").joinpath(f"prompt_assets/{asset}").read_text(encoding="utf-8").strip()


def build_system_prompt(prompt_method: str = "v1") -> str:
    method = load_reasoning_method(prompt_method)
    return f"""You solve ARC-AGI-2 abstract grid reasoning tasks. Infer a single transformation rule that is consistent with every provided demonstration. Think carefully before answering. After reasoning, return exactly one final grid inside <answer>...</answer>. The answer must be a JSON list of equally sized rows containing only integers 0 through 9.

Use the following reusable reasoning procedure when solving the task:

{method}"""


# Backward-compatible aliases for code/tests that imported the original constants.
STUDENT_UNIVERSAL_METHOD = load_reasoning_method("v1")
SYSTEM_PROMPT = build_system_prompt("v1")


def grid_to_text(grid: Grid) -> str:
    return json.dumps(grid, separators=(",", ":"))


def build_messages(
    support_pairs: list[Pair] | tuple[Pair, ...],
    query: Grid,
    *,
    prompt_method: str = "v1",
) -> list[dict[str, str]]:
    sections: list[str] = []
    for i, pair in enumerate(support_pairs, 1):
        sections.append(
            f"Demonstration {i}\nInput:\n{grid_to_text(pair.input)}\nOutput:\n{grid_to_text(pair.output)}"
        )
    sections.append(f"Test input:\n{grid_to_text(query)}")
    sections.append("Infer the rule and solve the test input.")
    return [
        {"role": "system", "content": build_system_prompt(prompt_method)},
        {"role": "user", "content": "\n\n".join(sections)},
    ]
