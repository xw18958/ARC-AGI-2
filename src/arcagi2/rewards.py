from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from .parsing import ANSWER_RE, parse_grid
from .types import Grid

# A same-shape prediction has made meaningful ARC progress even if no cell is correct yet.
# The remaining reward is proportional to exact cell accuracy. Exact correctness remains a
# separate, dominant reward in GRPOConfig.
_SHAPE_FLOOR = 0.25


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        item = completion[-1]
        if isinstance(item, dict):
            return str(item.get("content", ""))
    return str(completion)


def _same_shape(a: Grid, b: Grid) -> bool:
    return len(a) == len(b) and len(a[0]) == len(b[0])


def _cell_accuracy(predicted: Grid, truth: Grid) -> float:
    correct = sum(
        int(p == t)
        for pred_row, truth_row in zip(predicted, truth)
        for p, t in zip(pred_row, truth_row)
    )
    total = len(truth) * len(truth[0])
    return correct / total


def exact_grid_reward(completions, ground_truth, log_metric=None, **kwargs) -> list[float]:
    """Primary ARC reward: 1 only for an exact final-grid match.

    GRPO applies the resulting advantage to the complete generated trajectory, including the
    model's self-generated thinking. Diagnostics expose how often groups contain useful outcome
    variance versus all-wrong/all-correct rollouts.
    """
    predicted = [parse_grid(_completion_text(completion)) for completion in completions]
    truths = [json.loads(text) for text in ground_truth]
    rewards = [float(pred == truth) for pred, truth in zip(predicted, truths)]

    if log_metric and rewards:
        log_metric("arc_exact_grid", sum(rewards) / len(rewards))
        log_metric("arc_parseable", sum(p is not None for p in predicted) / len(predicted))

        task_ids = kwargs.get("task_id")
        target_indices = kwargs.get("target_index")
        shot_counts = kwargs.get("shot_count")
        if task_ids is not None and target_indices is not None and shot_counts is not None:
            grouped: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
            for task_id, target_index, shot_count, truth, reward in zip(
                task_ids, target_indices, shot_counts, ground_truth, rewards
            ):
                key = (str(task_id), int(target_index), int(shot_count), str(truth))
                grouped[key].append(reward)
            if grouped:
                values = list(grouped.values())
                all_wrong = sum(all(r == 0.0 for r in group) for group in values) / len(values)
                all_correct = sum(all(r == 1.0 for r in group) for group in values) / len(values)
                mixed = sum(len(set(group)) > 1 for group in values) / len(values)
                log_metric("arc_group_all_wrong", all_wrong)
                log_metric("arc_group_all_correct", all_correct)
                log_metric("arc_group_mixed", mixed)

    return rewards


def grid_progress_reward(completions, ground_truth, log_metric=None, **kwargs) -> list[float]:
    """Small dense, fully verifiable ARC progress reward in [0, 1].

    Invalid or wrong-shape outputs receive 0. A correct-shape grid receives a small floor plus
    cell-level exact accuracy. This gives GRPO relative signal before any rollout fully solves a
    hard query, while the configured weight keeps exact correctness dominant.
    """
    rewards: list[float] = []
    shape_matches = 0
    for completion, truth_text in zip(completions, ground_truth):
        predicted = parse_grid(_completion_text(completion))
        truth = json.loads(truth_text)
        if predicted is None or not _same_shape(predicted, truth):
            rewards.append(0.0)
            continue
        shape_matches += 1
        cell_accuracy = _cell_accuracy(predicted, truth)
        rewards.append(_SHAPE_FLOOR + (1.0 - _SHAPE_FLOOR) * cell_accuracy)

    if log_metric and rewards:
        log_metric("arc_progress", sum(rewards) / len(rewards))
        log_metric("arc_shape_match", shape_matches / len(rewards))
    return rewards


def answer_format_reward(completions, log_metric=None, **kwargs) -> list[float]:
    """Tiny format reward for an explicit, parseable <answer>...</answer> ARC grid."""
    rewards: list[float] = []
    for completion in completions:
        text = _completion_text(completion)
        explicit = bool(ANSWER_RE.search(text))
        rewards.append(float(explicit and parse_grid(text) is not None))
    if log_metric and rewards:
        log_metric("arc_answer_format", sum(rewards) / len(rewards))
    return rewards
