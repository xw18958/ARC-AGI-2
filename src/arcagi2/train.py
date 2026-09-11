from __future__ import annotations

import argparse
import math
import os

from .config import load_config
from .data import load_bundle
from .episodes import episode_count, training_row_stream
from .rewards import exact_grid_reward


def _make_dataset(tasks, seed: int):
    from datasets import IterableDataset

    # Streaming keeps augmentation on-the-fly and avoids materializing all support subsets.
    return IterableDataset.from_generator(
        training_row_stream,
        gen_kwargs={"tasks": tasks, "seed": seed},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen3 on ARC-AGI-2 with LoRA + GRPO")
    parser.add_argument("--config", default="configs/qwen3_8b_grpo_lora.yaml")
    parser.add_argument("--data", required=True, help="Competition ZIP or extracted dataset directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bundle = load_bundle(args.data)
    seed = int(cfg.get("seed", 42))
    train_dataset = _make_dataset(bundle.train, seed)
    logical_episodes = episode_count(bundle.train)

    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )

    t = cfg["training"]
    g = cfg["grpo"]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_device_effective_batch = (
        int(t["per_device_train_batch_size"]) * int(t["gradient_accumulation_steps"])
    )
    global_effective_batch = per_device_effective_batch * world_size
    num_generations = int(g["num_generations"])
    if global_effective_batch % num_generations != 0:
        raise ValueError(
            "WORLD_SIZE * per_device_train_batch_size * gradient_accumulation_steps "
            "must be divisible by num_generations."
        )

    # TRL repeats each unique prompt num_generations times inside the effective batch.
    # Thus this is the number of unique ARC episode specifications consumed per optimizer step.
    prompts_per_optimizer_step = max(1, global_effective_batch // num_generations)
    max_steps = math.ceil(
        logical_episodes * int(t.get("logical_epochs", 1)) / prompts_per_optimizer_step
    )
    if t.get("max_steps_override") is not None:
        max_steps = int(t["max_steps_override"])

    v = cfg.get("vllm", {})
    training_args = GRPOConfig(
        output_dir=t["output_dir"],
        max_steps=max_steps,
        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        learning_rate=float(t["learning_rate"]),
        warmup_ratio=float(t.get("warmup_ratio", 0.0)),
        weight_decay=float(t.get("weight_decay", 0.0)),
        max_grad_norm=float(t.get("max_grad_norm", 1.0)),
        bf16=bool(t.get("bf16", True)),
        gradient_checkpointing=bool(t.get("gradient_checkpointing", True)),
        logging_steps=int(t.get("logging_steps", 10)),
        save_steps=int(t.get("save_steps", 100)),
        save_total_limit=int(t.get("save_total_limit", 3)),
        report_to=t.get("report_to", "none"),
        seed=seed,
        remove_unused_columns=False,
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
        num_generations=num_generations,
        max_completion_length=int(g["max_completion_length"]),
        temperature=float(g.get("temperature", 0.6)),
        top_p=float(g.get("top_p", 0.95)),
        top_k=int(g.get("top_k", 20)),
        min_p=float(g.get("min_p", 0.0)),
        beta=float(g.get("beta", 0.0)),
        loss_type=g.get("loss_type", "dapo"),
        shuffle_dataset=bool(g.get("shuffle_dataset", False)),
        log_completions=bool(g.get("log_completions", True)),
        num_completions_to_print=int(g.get("num_completions_to_print", 2)),
        chat_template_kwargs={"enable_thinking": bool(cfg["model"].get("enable_thinking", True))},
        use_vllm=bool(v.get("enabled", False)),
        vllm_mode=v.get("mode", "colocate"),
        vllm_gpu_memory_utilization=float(v.get("gpu_memory_utilization", 0.3)),
    )

    trainer = GRPOTrainer(
        model=cfg["model"]["name_or_path"],
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=exact_grid_reward,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=t.get("resume_from_checkpoint") or None)
    trainer.save_model(t["output_dir"])


if __name__ == "__main__":
    main()
