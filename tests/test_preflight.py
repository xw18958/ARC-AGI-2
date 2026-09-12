from arcagi2.preflight import _max_prompt_tokens, _token_count
from arcagi2.train import _warmup_kwargs


def test_token_count_accepts_list_and_batch_encoding_shapes():
    assert _token_count([1, 2, 3]) == 3
    assert _token_count({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}) == 3
    assert _token_count({"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}) == 3


def test_max_prompt_tokens_counts_input_ids_in_mapping():
    class MappingTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            size = 7 if args[0] == "long" else 3
            return {"input_ids": list(range(size)), "attention_mask": [1] * size}

    stats = _max_prompt_tokens(
        MappingTokenizer(),
        [("short-task", "short"), ("long-task", "long")],
        enable_thinking=True,
    )

    assert stats == {"count": 2, "max_tokens": 7, "max_task_id": "long-task"}


def test_warmup_ratio_maps_across_training_arguments_versions():
    assert _warmup_kwargs({"warmup_ratio"}, 0.03) == {"warmup_ratio": 0.03}
    assert _warmup_kwargs({"warmup_steps"}, 0.03) == {"warmup_steps": 0.03}
