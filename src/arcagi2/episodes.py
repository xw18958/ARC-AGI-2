from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

from .parsing import canonical_grid
from .prompts import build_messages
from .types import ArcTask, Pair


@dataclass(frozen=True)
class EpisodeSpec:
    task_id: str
    target_index: int
    shot_count: int


def build_episode_specs(tasks: dict[str, ArcTask]) -> list[EpisodeSpec]:
    """Cover every target pair and every valid shot count exactly once per logical epoch."""
    specs: list[EpisodeSpec] = []
    for task_id, task in tasks.items():
        pair_count = len(task.known_pairs)
        if pair_count < 2:
            continue
        for target_index in range(pair_count):
            for shot_count in range(1, pair_count):
                specs.append(EpisodeSpec(task_id, target_index, shot_count))
    return specs


def sample_supports(
    task: ArcTask, spec: EpisodeSpec, rng: random.Random
) -> tuple[list[Pair], Pair]:
    pairs = task.known_pairs
    target = pairs[spec.target_index]
    available = [i for i in range(len(pairs)) if i != spec.target_index]
    support_indices = rng.sample(available, spec.shot_count)
    rng.shuffle(support_indices)
    return [pairs[i] for i in support_indices], target


def make_episode_row(task: ArcTask, spec: EpisodeSpec, rng: random.Random) -> dict:
    supports, target = sample_supports(task, spec, rng)
    return {
        "prompt": build_messages(supports, target.input),
        "ground_truth": canonical_grid(target.output),
        "task_id": spec.task_id,
        "shot_count": spec.shot_count,
        "target_index": spec.target_index,
    }


def _ordered_specs_for_cycle(
    tasks: dict[str, ArcTask],
    specs: list[EpisodeSpec],
    rng: random.Random,
    *,
    curriculum: str,
) -> list[EpisodeSpec]:
    order = list(specs)
    rng.shuffle(order)
    if curriculum == "none":
        return order
    if curriculum != "high_to_low_evidence":
        raise ValueError(f"Unknown shot curriculum: {curriculum}")

    # More demonstrations for the same task provide more evidence. Sort by the fraction of the
    # available supports being shown, while retaining random order inside equal-evidence bands.
    # The stable sort preserves the preceding random shuffle for ties.
    def evidence_fraction(spec: EpisodeSpec) -> float:
        max_shots = len(tasks[spec.task_id].known_pairs) - 1
        return spec.shot_count / max_shots

    order.sort(key=evidence_fraction, reverse=True)
    return order


def training_row_stream(
    tasks: dict[str, ArcTask],
    seed: int = 42,
    curriculum: str = "high_to_low_evidence",
    curriculum_cycles: int = 1,
) -> Iterator[dict]:
    """Infinite on-the-fly augmentation stream.

    Each logical cycle visits every (task, target, shot-count) specification exactly once.
    Supports and their order are re-sampled each cycle. During the configured initial curriculum
    cycles, higher-evidence conditions are consumed before lower-evidence conditions; later cycles
    are fully shuffled. This is an evidence curriculum, not an assumption that tasks with more
    demonstrations are intrinsically easier.
    """
    specs = build_episode_specs(tasks)
    cycle = 0
    while True:
        cycle_rng = random.Random(seed + cycle * 1_000_003)
        cycle_curriculum = curriculum if cycle < curriculum_cycles else "none"
        order = _ordered_specs_for_cycle(tasks, specs, cycle_rng, curriculum=cycle_curriculum)
        for index, spec in enumerate(order):
            row_rng = random.Random(seed + cycle * 1_000_003 + index * 9_176 + 17)
            yield make_episode_row(tasks[spec.task_id], spec, row_rng)
        cycle += 1


def episode_count(tasks: dict[str, ArcTask]) -> int:
    return len(build_episode_specs(tasks))


def validation_cases(tasks: dict[str, ArcTask]) -> list[dict]:
    """Fixed cumulative-prefix validation: A, A+B, A+B+C, ... -> original test query."""
    cases: list[dict] = []
    for task_id, task in tasks.items():
        if task.test_outputs is None:
            raise ValueError(f"Validation solutions missing for {task_id}")
        for shot_count in range(1, len(task.train_pairs) + 1):
            supports = list(task.train_pairs[:shot_count])
            for query_index, (query, target) in enumerate(zip(task.test_inputs, task.test_outputs)):
                cases.append(
                    {
                        "task_id": task_id,
                        "shot_count": shot_count,
                        "max_shots": len(task.train_pairs),
                        "query_index": query_index,
                        "prompt": build_messages(supports, query),
                        "ground_truth": canonical_grid(target),
                    }
                )
    return cases
