# ARC-AGI-2: Qwen3-8B + LoRA + GRPO

A clean, scalable training and validation pipeline for the ARC Prize 2026 ARC-AGI-2 task.

## Design

- **Model:** `Qwen/Qwen3-8B`, thinking mode enabled.
- **Adaptation:** LoRA + GRPO; no labelled chain-of-thought is required.
- **Training:** 1,000 official training tasks only.
- **Dynamic augmentation:** every known pair can be the query; every valid shot count is covered; support examples and order are re-sampled on the fly.
- **Reward:** binary exact final-grid match; reward-sparsity diagnostics are logged.
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
│   ├── preflight.py
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

## Audit before training

The dataset checker reports split statistics, train/evaluation leakage checks, local-test duplication, and augmentation weighting:

```bash
arc-stats --data /path/to/arc-prize-2026-arc-agi-2.zip
```

Then run the model/runtime preflight. It loads the Qwen tokenizer/config but not the 8B weights, checks the installed TRL API, measures the longest ARC prompts, verifies prompt + completion fits the model context, and reports expected rollout counts:

```bash
arc-preflight \
  --config configs/qwen3_8b_grpo_lora.yaml \
  --data /path/to/arc-prize-2026-arc-agi-2.zip
```

For the supplied competition ZIP, the audited experiment contains **15,350** target×shot training episode specifications per logical cycle and **523** fixed validation query×shot cases.

## Train

Single/multi-GPU launch is handled by Accelerate:

```bash
accelerate launch -m arcagi2.train \
  --config configs/qwen3_8b_grpo_lora.yaml \
  --data /path/to/arc-prize-2026-arc-agi-2.zip
```

The config exposes LoRA, GRPO group size, optimizer, generation, checkpointing, and optional vLLM settings. The episode stream already shuffles each logical cycle, so TRL's extra iterable-dataset shuffle is disabled to preserve complete target×shot coverage.

GRPO logs `arc_exact_grid`, `arc_parseable`, `arc_group_all_wrong`, `arc_group_all_correct`, and `arc_group_mixed`. If almost every group is all-wrong, exact-match-only GRPO is receiving little relative learning signal and the next experiment should address reward sparsity rather than silently continuing a long run.

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

The lightweight tests exercise augmentation coverage, answer parsing, candidate voting, exact-grid reward diagnostics, and the shot-efficiency aggregation without loading Qwen3-8B.
