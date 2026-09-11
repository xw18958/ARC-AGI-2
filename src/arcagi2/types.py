from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

Grid: TypeAlias = list[list[int]]


@dataclass(frozen=True)
class Pair:
    input: Grid
    output: Grid


@dataclass(frozen=True)
class ArcTask:
    task_id: str
    train_pairs: tuple[Pair, ...]
    test_inputs: tuple[Grid, ...]
    test_outputs: tuple[Grid, ...] | None = None

    @property
    def known_pairs(self) -> tuple[Pair, ...]:
        """All labelled pairs available for training augmentation."""
        pairs = list(self.train_pairs)
        if self.test_outputs is not None:
            if len(self.test_inputs) != len(self.test_outputs):
                raise ValueError(f"Mismatched test inputs/outputs for task {self.task_id}")
            pairs.extend(Pair(inp, out) for inp, out in zip(self.test_inputs, self.test_outputs))
        return tuple(pairs)
