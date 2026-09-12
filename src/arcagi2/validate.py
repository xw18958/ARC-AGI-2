from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import load_bundle
from .episodes import validation_cases
from .metrics import ValidationRecord, choose_attempts, solved_by_attempts, summarize_validation


def _load_model(model_cfg: dict, checkpoint: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = model_cfg["name_or_path"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    if checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()
    return model, tokenizer


def _batched(items: list[dict], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _generate_batch(model, tokenizer, cases: list[dict], cfg: dict, *, enable_thinking: bool) -> list[list[str]]:
    import torch

    rendered = [
        tokenizer.apply_chat_template(
            case["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        for case in cases
    ]
    encoded = tokenizer(rendered, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    num_candidates = int(cfg["num_candidates"])
    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            do_sample=True,
            temperature=float(cfg.get("temperature", 0.6)),
            top_p=float(cfg.get("top_p", 0.95)),
            top_k=int(cfg.get("top_k", 20)),
            max_new_tokens=int(cfg.get("max_new_tokens", 4096)),
            num_return_sequences=num_candidates,
        )
    prompt_len = encoded["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
    return [decoded[i * num_candidates : (i + 1) * num_candidates] for i in range(len(cases))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed cumulative-shot ARC validation")
    parser.add_argument("--config", default="configs/qwen3_8b_grpo_lora.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", default=None, help="LoRA adapter checkpoint; omit for base Qwen")
    parser.add_argument("--output", default="validation_results.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    from transformers import set_seed

    set_seed(int(cfg.get("seed", 42)))
    bundle = load_bundle(args.data)
    val_cfg = cfg["validation"]
    if not bool(val_cfg.get("fixed_prefix", True)):
        raise ValueError("Only the fixed cumulative-prefix validation protocol is implemented.")
    prompt_method = cfg.get("prompt", {}).get("method", "v1")
    cases = validation_cases(bundle.evaluation, prompt_method=prompt_method)
    model, tokenizer = _load_model(cfg["model"], args.checkpoint)
    enable_thinking = bool(cfg["model"].get("enable_thinking", True))
    records: list[ValidationRecord] = []
    details: list[dict] = []

    for batch in _batched(cases, int(val_cfg["batch_size"])):
        candidate_groups = _generate_batch(model, tokenizer, batch, val_cfg, enable_thinking=enable_thinking)
        for case, candidates in zip(batch, candidate_groups):
            attempts = choose_attempts(candidates, int(val_cfg.get("num_attempts", 2)))
            solved = solved_by_attempts(attempts, case["ground_truth"])
            records.append(
                ValidationRecord(
                    task_id=case["task_id"], shot_count=case["shot_count"], max_shots=case["max_shots"],
                    query_index=case["query_index"], solved=solved,
                )
            )
            details.append(
                {
                    "task_id": case["task_id"], "shot_count": case["shot_count"],
                    "query_index": case["query_index"], "ground_truth": json.loads(case["ground_truth"]),
                    "attempts": attempts, "solved": solved,
                }
            )

    summary = summarize_validation(records)
    payload = {"prompt_method": prompt_method, "summary": summary, "cases": details}
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
