from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from .parsing import parse_grid


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        item = completion[-1]
        if isinstance(item, dict):
            return str(item.get("content", ""))
    return str(completion)


def exact_grid_reward(completions, ground_truth, log_metric=None, **kwargs) -> list[float]:
    """Binary ARC reward: 1 only for an exact final-grid match.

    Besides the reward itself, log diagnostics that are important for binary-reward GRPO:
    parseable output rate and the fractions of all-wrong/all-correct/mixed prompt groups.
    A mixed group is the useful case for standard relative GRPO because it contains reward variance.
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
