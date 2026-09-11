from __future__ import annotations

import json

from .types import Grid, Pair

SYSTEM_PROMPT = """You solve ARC-AGI-2 abstract grid reasoning tasks. Infer a single transformation rule that is consistent with every provided demonstration. Think carefully before answering. After reasoning, return exactly one final grid inside <answer>...</answer>. The answer must be a JSON list of equally sized rows containing only integers 0 through 9."""


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
