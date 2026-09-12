from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from importlib.metadata import version

from .config import load_config
from .data import load_bundle
from .episodes import build_episode_specs, validation_cases
from .prompts import build_messages
from .train import _warmup_kwargs

_REQUIRED_GRPO_FIELDS = {
    "output_dir",
    "max_steps",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "weight_decay",
    "max_grad_norm",
    "bf16",
    "gradient_checkpointing",
    "logging_steps",
    "save_steps",
    "save_total_limit",
    "report_to",
    "seed",
    "remove_unused_columns",
    "trust_remote_code",
    "num_generations",
    "max_completion_length",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "beta",
    "loss_type",
    "scale_rewards",
    "shuffle_dataset",
    "chat_template_kwargs",
    "reward_weights",
    "mask_truncated_completions",
    "log_completions",
    "num_completions_to_print",
    "use_vllm",
    "vllm_mode",
    "vllm_gpu_memory_utilization",
}


def _token_count(tokenized) -> int:
    """Count one rendered prompt across tokenizer return types.

    Recent Transformers releases return a ``BatchEncoding`` from
    ``apply_chat_template`` even for one unbatched conversation. Counting that
    mapping directly reports its number of fields instead of its tokens.
    """
    input_ids = tokenized.get("input_ids") if isinstance(tokenized, Mapping) else tokenized
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        return int(shape[-1])
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        if len(input_ids) != 1:
            raise ValueError("Expected one tokenized prompt")
        input_ids = input_ids[0]
    return len(input_ids)


def _full_shot_training_prompts(tasks):
    """Yield the longest support condition for every possible training target."""
    for task in tasks.values():
        pairs = task.known_pairs
        if len(pairs) < 2:
            continue
        for target_index, target in enumerate(pairs):
            supports = [pair for i, pair in enumerate(pairs) if i != target_index]
            yield task.task_id, build_messages(supports, target.input)


def _max_prompt_tokens(tokenizer, prompts, *, enable_thinking: bool):
    maximum = 0
    maximum_id = None
    count = 0
    for task_id, messages in prompts:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        length = _token_count(token_ids)
        count += 1
        if length > maximum:
            maximum = length
            maximum_id = task_id
    return {"count": count, "max_tokens": maximum, "max_task_id": maximum_id}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight ARC/Qwen/TRL before an expensive run")
    parser.add_argument("--config", default="configs/qwen3_8b_grpo_lora.yaml")
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    bundle = load_bundle(args.data)

    from transformers import AutoConfig, AutoTokenizer
    from trl import GRPOConfig

    grpo_fields = set(GRPOConfig.__dataclass_fields__)
    missing = sorted(_REQUIRED_GRPO_FIELDS - grpo_fields)
    if missing:
        raise RuntimeError(f"Installed TRL GRPOConfig is missing required fields: {missing}")
    warmup_kwargs = _warmup_kwargs(grpo_fields, float(cfg["training"].get("warmup_ratio", 0.0)))

    rewards = cfg["rewards"]
    reward_weights = [
        float(rewards["exact_weight"]),
        float(rewards["progress_weight"]),
        float(rewards["format_weight"]),
    ]
    if reward_weights[0] <= sum(max(0.0, w) for w in reward_weights[1:]):
        raise RuntimeError("Exact reward is not dominant over auxiliary shaping rewards")

    model_cfg = cfg["model"]
    model_name = model_cfg["name_or_path"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    enable_thinking = bool(model_cfg.get("enable_thinking", True))
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    hf_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    train_prompt_stats = _max_prompt_tokens(
        tokenizer,
        _full_shot_training_prompts(bundle.train),
        enable_thinking=enable_thinking,
    )
    val_cases = validation_cases(bundle.evaluation)
    val_prompt_stats = _max_prompt_tokens(
        tokenizer,
        ((case["task_id"], case["prompt"]) for case in val_cases),
        enable_thinking=enable_thinking,
    )

    context = int(getattr(hf_cfg, "max_position_embeddings", 0) or 0)
    completion = int(cfg["grpo"]["max_completion_length"])
    longest_prompt = max(train_prompt_stats["max_tokens"], val_prompt_stats["max_tokens"])
    if context and longest_prompt + completion > context:
        raise RuntimeError(
            f"Prompt+completion can exceed model context: {longest_prompt}+{completion}>{context}"
        )

    logical_episodes = len(build_episode_specs(bundle.train))
    num_generations = int(cfg["grpo"]["num_generations"])
    val_candidates = int(cfg["validation"]["num_candidates"])
    report = {
        "versions": {
            name: version(name)
            for name in ("torch", "transformers", "datasets", "trl", "peft", "accelerate")
        },
        "model": {
            "name": model_name,
            "context_tokens": context,
            "enable_thinking": enable_thinking,
        },
        "augmentation": cfg["augmentation"],
        "rewards": {
            "weights": {
                "exact": reward_weights[0],
                "progress": reward_weights[1],
                "format": reward_weights[2],
            },
            "exact_is_dominant": True,
        },
        "training": {
            "logical_episode_specs": logical_episodes,
            "rollouts_per_logical_cycle": logical_episodes * num_generations,
            "num_generations": num_generations,
            "max_completion_length": completion,
            "mask_truncated_completions": bool(
                cfg["grpo"].get("mask_truncated_completions", True)
            ),
            "max_full_shot_prompt": train_prompt_stats,
        },
        "validation": {
            "cases": len(val_cases),
            "candidate_generations_per_run": len(val_cases) * val_candidates,
            "num_candidates": val_candidates,
            "max_prompt": val_prompt_stats,
        },
        "context_margin_tokens": context - longest_prompt - completion if context else None,
        "grpo_api_check": {
            "status": "ok",
            "warmup_argument": next(iter(warmup_kwargs)),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
