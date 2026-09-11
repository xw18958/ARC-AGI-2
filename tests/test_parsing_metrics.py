from arcagi2.metrics import ValidationRecord, choose_attempts, summarize_validation
from arcagi2.parsing import parse_grid


def test_parse_answer_block_after_reasoning():
    text = "<think>I considered [[9]].</think><answer>[[1,0],[0,1]]</answer>"
    assert parse_grid(text) == [[1, 0], [0, 1]]


def test_candidate_voting_returns_distinct_top_two():
    candidates = [
        "<answer>[[1]]</answer>",
        "<answer>[[2]]</answer>",
        "<answer>[[1]]</answer>",
        "bad",
    ]
    assert choose_attempts(candidates, 2) == [[[1]], [[2]]]


def test_shot_efficiency_macro_weights_tasks_equally():
    records = [
        # Task A: 2 shots, scores 1 then 0 -> task score .5
        ValidationRecord("A", 1, 2, 0, True),
        ValidationRecord("A", 2, 2, 0, False),
        # Task B: 3 shots, scores 1,1,1 -> task score 1
        ValidationRecord("B", 1, 3, 0, True),
        ValidationRecord("B", 2, 3, 0, True),
        ValidationRecord("B", 3, 3, 0, True),
    ]
    metrics = summarize_validation(records)
    assert metrics["shot_efficiency"] == 0.75
    assert metrics["full_shot_2attempt_accuracy"] == 0.5
