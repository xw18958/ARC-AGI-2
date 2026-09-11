from __future__ import annotations

import json
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import ArcTask, Grid, Pair

CHALLENGE_FILES = {
    "train": "arc-agi_training_challenges.json",
    "evaluation": "arc-agi_evaluation_challenges.json",
    "test": "arc-agi_test_challenges.json",
}
SOLUTION_FILES = {
    "train": "arc-agi_training_solutions.json",
    "evaluation": "arc-agi_evaluation_solutions.json",
}


@dataclass(frozen=True)
class ArcBundle:
    train: dict[str, ArcTask]
    evaluation: dict[str, ArcTask]
    test: dict[str, ArcTask]


def _validate_grid(grid: Any, *, where: str) -> Grid:
    if not isinstance(grid, list) or not grid or not all(isinstance(row, list) for row in grid):
        raise ValueError(f"Invalid grid at {where}")
    width = len(grid[0])
    if width == 0 or any(len(row) != width for row in grid):
        raise ValueError(f"Ragged/empty grid at {where}")
    if len(grid) > 30 or width > 30:
        raise ValueError(f"Grid exceeds 30x30 at {where}")
    for row in grid:
        for value in row:
            if type(value) is not int or not 0 <= value <= 9:
                raise ValueError(f"Non-ARC cell value at {where}: {value!r}")
    return grid


def _read_json(path: Path, filename: str) -> dict[str, Any]:
    return json.loads((path / filename).read_text())


def _read_zip_json(path: Path, filename: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        matches = [n for n in zf.namelist() if not n.startswith("__MACOSX/") and n.endswith(filename)]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one {filename} in {path}; found {matches}")
        with zf.open(matches[0]) as f:
            return json.load(f)


def _load_json(data_path: Path, filename: str) -> dict[str, Any]:
    if data_path.is_file() and data_path.suffix.lower() == ".zip":
        return _read_zip_json(data_path, filename)
    if data_path.is_dir():
        direct = data_path / filename
        if direct.exists():
            return _read_json(data_path, filename)
        matches = [p for p in data_path.rglob(filename) if "__MACOSX" not in p.parts]
        if len(matches) == 1:
            return json.loads(matches[0].read_text())
    raise FileNotFoundError(f"Could not find {filename} under {data_path}")


def _parse_tasks(
    challenges: dict[str, Any], solutions: dict[str, Any] | None = None
) -> dict[str, ArcTask]:
    tasks: dict[str, ArcTask] = {}
    for task_id, task in challenges.items():
        train_pairs = tuple(
            Pair(
                _validate_grid(p["input"], where=f"{task_id}.train.input"),
                _validate_grid(p["output"], where=f"{task_id}.train.output"),
            )
            for p in task["train"]
        )
        test_inputs = tuple(
            _validate_grid(p["input"], where=f"{task_id}.test.input") for p in task["test"]
        )
        test_outputs = None
        if solutions is not None:
            outputs = solutions[task_id]
            test_outputs = tuple(
                _validate_grid(g, where=f"{task_id}.test.output") for g in outputs
            )
            if len(test_inputs) != len(test_outputs):
                raise ValueError(f"Solution count mismatch for {task_id}")
        tasks[task_id] = ArcTask(task_id, train_pairs, test_inputs, test_outputs)
    return tasks


def load_bundle(data_path: str | Path) -> ArcBundle:
    """Load the competition ZIP or an extracted dataset directory.

    The local `test` file is loaded only for inspection/submission plumbing. Training code
    intentionally uses only `train`; validation uses only `evaluation`.
    """
    path = Path(data_path).expanduser().resolve()
    train_ch = _load_json(path, CHALLENGE_FILES["train"])
    train_sol = _load_json(path, SOLUTION_FILES["train"])
    eval_ch = _load_json(path, CHALLENGE_FILES["evaluation"])
    eval_sol = _load_json(path, SOLUTION_FILES["evaluation"])
    test_ch = _load_json(path, CHALLENGE_FILES["test"])
    return ArcBundle(
        train=_parse_tasks(train_ch, train_sol),
        evaluation=_parse_tasks(eval_ch, eval_sol),
        test=_parse_tasks(test_ch, None),
    )


def split_statistics(tasks: dict[str, ArcTask], *, include_known_test: bool = False) -> dict[str, Any]:
    shot_counts = Counter(len(t.train_pairs) for t in tasks.values())
    test_query_counts = Counter(len(t.test_inputs) for t in tasks.values())
    all_grids: list[Grid] = []
    for task in tasks.values():
        pairs = task.known_pairs if include_known_test else task.train_pairs
        for pair in pairs:
            all_grids.extend([pair.input, pair.output])
        all_grids.extend(task.test_inputs)
    cells = [len(g) * len(g[0]) for g in all_grids]
    colors = [len({v for row in g for v in row}) for g in all_grids]
    cells_sorted = sorted(cells)
    colors_sorted = sorted(colors)

    def median(xs: list[int]) -> float:
        n = len(xs)
        return float(xs[n // 2]) if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

    return {
        "tasks": len(tasks),
        "demonstration_pairs": sum(len(t.train_pairs) for t in tasks.values()),
        "test_queries": sum(len(t.test_inputs) for t in tasks.values()),
        "shot_distribution": dict(sorted(shot_counts.items())),
        "test_query_distribution": dict(sorted(test_query_counts.items())),
        "median_grid_cells": median(cells_sorted),
        "median_grid_colors": median(colors_sorted),
    }
