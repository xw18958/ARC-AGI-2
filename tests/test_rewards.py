from arcagi2.rewards import answer_format_reward, exact_grid_reward, grid_progress_reward


def test_exact_reward_and_group_sparsity_metrics():
    metrics = {}

    def log_metric(name, value):
        metrics[name] = value

    rewards = exact_grid_reward(
        completions=[
            "<answer>[[1]]</answer>",
            "<answer>[[0]]</answer>",
            "<answer>[[0]]</answer>",
            "not a grid",
        ],
        ground_truth=["[[1]]", "[[1]]", "[[2]]", "[[2]]"],
        task_id=["A", "A", "B", "B"],
        target_index=[0, 0, 1, 1],
        shot_count=[1, 1, 2, 2],
        log_metric=log_metric,
    )
    assert rewards == [1.0, 0.0, 0.0, 0.0]
    assert metrics["arc_exact_grid"] == 0.25
    assert metrics["arc_parseable"] == 0.75
    assert metrics["arc_group_mixed"] == 0.5
    assert metrics["arc_group_all_wrong"] == 0.5
    assert metrics["arc_group_all_correct"] == 0.0


def test_progress_reward_is_dense_but_bounded():
    rewards = grid_progress_reward(
        completions=[
            "<answer>[[1,2],[3,4]]</answer>",   # exact -> 1
            "<answer>[[1,2],[0,0]]</answer>",   # same shape, 2/4 cells correct
            "<answer>[[1,2,3]]</answer>",       # wrong shape -> 0
            "bad",
        ],
        ground_truth=["[[1,2],[3,4]]"] * 4,
    )
    assert rewards[0] == 1.0
    assert 0.25 < rewards[1] < 1.0
    assert rewards[2:] == [0.0, 0.0]


def test_truncated_thinking_cannot_earn_progress_from_mentioned_grids():
    completion = "<think>Still reasoning from [[1,2],[3,4]]"
    assert grid_progress_reward([completion], ["[[1,2],[3,4]]"]) == [0.0]
    assert answer_format_reward([completion]) == [0.0]


def test_format_reward_requires_explicit_answer_block():
    rewards = answer_format_reward(
        completions=[
            "<think>x</think><answer>[[1]]</answer>",
            "<think>x</think>[[1]]",
            "<answer>not-json</answer>",
        ]
    )
    assert rewards == [1.0, 0.0, 0.0]


def test_exact_reward_dominates_auxiliary_weights():
    exact_weight, progress_weight, format_weight = 1.0, 0.20, 0.02
    best_incorrect = progress_weight * 1.0 + format_weight * 1.0
    worst_exact = exact_weight * 1.0
    assert worst_exact > best_incorrect
