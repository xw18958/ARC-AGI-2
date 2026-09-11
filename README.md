# ARC-AGI-2: Qwen3-8B + LoRA + GRPO

A clean, scalable training and validation pipeline for the ARC Prize 2026 ARC-AGI-2 task.

## Design

- **Model:** `Qwen/Qwen3-8B`, thinking mode enabled.
- **Adaptation:** LoRA + GRPO; no labelled chain-of-thought is required.
- **Training:** 1,000 official training tasks only.
- **Dynamic augmentation:** every known pair can be the query; every valid shot count is covered; support examples and order are re-sampled on the fly.
- **Reward:** binary exact final-grid match.
- **Validation:** fixed 120 official evaluation tasks, evaluated at cumulative prefixes `A`, `A+B`, `A+B+C`, ... against the original test query(s).
- **Checkpoint metric:** task-macro shot-efficiency score.
- **Competition diagnostic:** full-shot two-attempt exact-match accuracy.
- **Local test file:** deliberately excluded from train/validation development.

See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) for the full experimental contract.

## Repository layout

```text
ARC-AGI-2/
├── configs/
│   └── qwen3_8b_grpo_lora.yaml
├── docs/
│   └── IMPLEMENTATION_PLAN.md
├── src/arcagi2/
│   ├── config.py
│   ├── data.py
│   ├── episodes.py
│   ├── inspect_data.py
│   ├── metrics.py
│   ├── parsing.py
│   ├── prompts.py
│   ├── rewards.py
│   ├── train.py
│   ├── types.py
│   └── validate.py
├── tests/
├── pyproject.toml
└── README.md
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

For optional vLLM rollout acceleration:

```bash
pip install -e '.[vllm]'
```

For W&B logging:

```bash
pip install -e '.[wandb]'
```

## Dataset check

The loader accepts either the Kaggle ZIP directly or an extracted directory:

```bash
arc-stats --data /path/to/arc-prize-2026-arc-agi-2.zip
```

## Train

Single/multi-GPU launch is handled by Accelerate:

```bash
accelerate launch -m arcagi2.train \
  --config configs/qwen3_8b_grpo_lora.yaml \
  --data /path/to/arc-prize-2026-arc-agi-2.zip
```

The config exposes LoRA, GRPO group size, optimizer, generation, checkpointing, and optional vLLM settings.

## Validate

Base model:

```bash
arc-validate \
  --config configs/qwen3_8b_grpo_lora.yaml \
  --data /path/to/arc-prize-2026-arc-agi-2.zip
```

LoRA checkpoint:

```bash
arc-validate \
  --config configs/qwen3_8b_grpo_lora.yaml \
  --data /path/to/arc-prize-2026-arc-agi-2.zip \
  --checkpoint outputs/qwen3_8b_grpo_lora/checkpoint-100
```

The validator writes a JSON report containing the overall shot-efficiency score, per-shot accuracies, full-shot two-attempt accuracy, and every evaluated case.

## Tests

```bash
pytest -q
```

The unit tests exercise augmentation coverage, answer parsing, candidate voting, and the shot-efficiency aggregation without loading Qwen3-8B.
