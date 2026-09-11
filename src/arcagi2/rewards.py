from __future__ import annotations

import json
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
    """Binary ARC reward: 1 only for an exact final-grid match."""
    rewards: list[float] = []
    for completion, truth_text in zip(completions, ground_truth):
        predicted = parse_grid(_completion_text(completion))
        truth = json.loads(truth_text)
        rewards.append(float(predicted == truth))
    if log_metric and rewards:
        log_metric("arc_exact_grid", sum(rewards) / len(rewards))
    return rewards
