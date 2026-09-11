import random

from arcagi2.episodes import (
    build_episode_specs,
    sample_supports,
    training_row_stream,
    validation_cases,
)
from arcagi2.types import ArcTask, Pair


def grid(v):
    return [[v]]


def make_task():
    return ArcTask(
        "t",
        (Pair(grid(0), grid(1)), Pair(grid(2), grid(3))),
        (grid(4),),
        (grid(5),),
    )


def test_every_target_and_every_shot_is_covered():
    task = make_task()
    specs = build_episode_specs({"t": task})
    # 3 known pairs -> 3 targets * {1-shot, 2-shot}
    assert len(specs) == 6
    assert {(s.target_index, s.shot_count) for s in specs} == {
        (0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)
    }


def test_target_never_leaks_into_supports():
    task = make_task()
    spec = build_episode_specs({"t": task})[-1]
    supports, target = sample_supports(task, spec, random.Random(1))
    assert len(supports) == spec.shot_count
    assert target not in supports


def test_high_to_low_evidence_curriculum_preserves_full_coverage():
    task = make_task()
    tasks = {"t": task}
    rows = training_row_stream(tasks, seed=3, curriculum="high_to_low_evidence", curriculum_cycles=1)
    first_cycle = [next(rows) for _ in range(6)]
    assert [row["shot_count"] for row in first_cycle[:3]] == [2, 2, 2]
    assert [row["shot_count"] for row in first_cycle[3:]] == [1, 1, 1]
    assert {(r["target_index"], r["shot_count"]) for r in first_cycle} == {
        (0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)
    }


def test_validation_uses_fixed_original_prefixes():
    task = make_task()
    cases = validation_cases({"t": task})
    assert [c["shot_count"] for c in cases] == [1, 2]
    assert all(c["query_index"] == 0 for c in cases)
