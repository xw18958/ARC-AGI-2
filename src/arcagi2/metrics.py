from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean

from .parsing import canonical_grid, parse_grid
from .types import Grid


@dataclass(frozen=True)
class ValidationRecord:
    task_id: str
    shot_count: int
    max_shots: int
    query_index: int
    solved: bool


def choose_attempts(candidate_texts: list[str], num_attempts: int = 2) -> list[Grid | None]:
    """Vote over sampled final grids and return the top distinct candidates."""
    parsed = [parse_grid(text) for text in candidate_texts]
    valid = [(i, g) for i, g in enumerate(parsed) if g is not None]
    if not valid:
        return [None] * num_attempts

    counts = Counter(canonical_grid(g) for _, g in valid)
    first_index: dict[str, int] = {}
    grid_by_key: dict[str, Grid] = {}
    for i, grid in valid:
        key = canonical_grid(grid)
        first_index.setdefault(key, i)
        grid_by_key[key] = grid
    ranked = sorted(counts, key=lambda key: (-counts[key], first_index[key]))
    attempts: list[Grid | None] = [grid_by_key[key] for key in ranked[:num_attempts]]
    while len(attempts) < num_attempts:
        attempts.append(attempts[0] if attempts else None)
    return attempts


def solved_by_attempts(attempts: list[Grid | None], ground_truth_json: str) -> bool:
    truth = json.loads(ground_truth_json)
    return any(attempt == truth for attempt in attempts if attempt is not None)


def summarize_validation(records: list[ValidationRecord]) -> dict[str, float]:
    """Compute the user's shot-efficiency metric plus useful diagnostics.

    1) Average query correctness within each task and shot level.
    2) Average those shot-level accuracies within the task.
    3) Macro-average task scores so every ARC task has equal weight.
    """
    per_task_shot: dict[tuple[str, int], list[bool]] = defaultdict(list)
    max_shots: dict[str, int] = {}
    for r in records:
        per_task_shot[(r.task_id, r.shot_count)].append(r.solved)
        max_shots[r.task_id] = r.max_shots

    task_scores: dict[str, float] = {}
    per_shot_values: dict[int, list[float]] = defaultdict(list)
    for task_id, n_shots in max_shots.items():
        shot_scores = []
        for k in range(1, n_shots + 1):
            vals = per_task_shot[(task_id, k)]
            if not vals:
                raise ValueError(f"Missing validation results for {task_id}, {k}-shot")
            score = mean(float(v) for v in vals)
            shot_scores.append(score)
            per_shot_values[k].append(score)
        task_scores[task_id] = mean(shot_scores)

    full_shot_query_results = [r.solved for r in records if r.shot_count == r.max_shots]
    metrics = {
        "shot_efficiency": mean(task_scores.values()) if task_scores else 0.0,
        "full_shot_2attempt_accuracy": (
            mean(float(v) for v in full_shot_query_results) if full_shot_query_results else 0.0
        ),
    }
    for k, vals in sorted(per_shot_values.items()):
        metrics[f"{k}_shot_task_macro_accuracy"] = mean(vals)
    return metrics
