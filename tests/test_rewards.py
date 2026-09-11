from arcagi2.rewards import exact_grid_reward


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
