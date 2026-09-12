from __future__ import annotations

import json
from importlib.resources import files

from .types import Grid, Pair

STUDENT_UNIVERSAL_METHOD = (
    files("arcagi2")
    .joinpath("prompt_assets/student_universal_method.md")
    .read_text(encoding="utf-8")
    .strip()
)

SYSTEM_PROMPT = f"""You solve ARC-AGI-2 abstract grid reasoning tasks. Infer a single transformation rule that is consistent with every provided demonstration. Think carefully before answering. After reasoning, return exactly one final grid inside <answer>...</answer>. The answer must be a JSON list of equally sized rows containing only integers 0 through 9.

Use the following reusable reasoning procedure when solving the task:

{STUDENT_UNIVERSAL_METHOD}"""


def grid_to_text(grid: Grid) -> str:
    return json.dumps(grid, separators=(",", ":"))


def build_messages(support_pairs: list[Pair] | tuple[Pair, ...], query: Grid) -> list[dict[str, str]]:
    sections: list[str] = []
    for i, pair in enumerate(support_pairs, 1):
        sections.append(
            f"Demonstration {i}\nInput:\n{grid_to_text(pair.input)}\nOutput:\n{grid_to_text(pair.output)}"
        )
    sections.append(f"Test input:\n{grid_to_text(query)}")
    sections.append("Infer the rule and solve the test input.")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(sections)},
    ]
