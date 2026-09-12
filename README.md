# ARC-AGI-2: Qwen3-8B + LoRA + GRPO

A clean, scalable training and validation pipeline for the ARC Prize 2026 ARC-AGI-2 task.

## Design

- **Model:** `Qwen/Qwen3-8B`, native thinking mode enabled.
- **Adaptation:** LoRA + GRPO; no labelled chain-of-thought dataset is required.
- **Training:** 1,000 official training tasks only.
- **Dynamic augmentation:** every known pair can be the query; every valid shot count is covered; support examples and order are re-sampled on the fly.
- **Reasoning refinement:** exact-grid reward remains dominant, with small verifiable grid-progress and strict-format rewards to reduce sparse-reward dead zones.
- **Selectable reasoning prompt:** `v1` preserves the original empirical ARC guide; `v2` uses a compact decision controller while retaining the specialized method families in compressed trigger-based form.
- **Evidence curriculum:** the initial logical cycle can present high-evidence shot conditions before low-evidence conditions without changing target×shot coverage.
- **Validation:** fixed 120 official evaluation tasks at cumulative prefixes `A`, `A+B`, `A+B+C`, ... against the original test query(s).
- **Checkpoint metric:** task-macro shot-efficiency score.
- **Competition diagnostic:** full-shot two-attempt exact-match accuracy.
- **Local test file:** deliberately excluded from train/validation development.

See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) for the experimental contract, [`docs/REASONING_REFINEMENT.md`](docs/REASONING_REFINEMENT.md) for the CoT/GRPO refinement design, and [`docs/PROMPT_AB_TEST_PLAN.md`](docs/PROMPT_AB_TEST_PLAN.md) for the controlled V1-vs-V2 prompt gate before further GRPO work.

## Prompt versions

The prompt is selected in config:

```yaml
prompt:
  method: v2
```

Available methods:

- `v1`: `src/arcagi2/prompt_assets/student_universal_method.md` — original empirically derived ARC method guide.
- `v2`: `src/arcagi2/prompt_assets/student_universal_method_v2.md` — compact controller that explicitly treats color IDs as categorical symbols, tests simple substitution early, uses trigger-gated specialized methods, and stops once a rule explains every demonstration.

The code default remains `v1` for backward compatibility, while the supplied main and smoke configs explicitly select `v2`. Training, validation, and preflight use the same configured prompt path.

Do not infer a new completion-token limit from the first failed 4,096-token probe. Prompt V1 vs V2 must first be compared under identical generation settings; any future completion cap should be based on observed natural termination lengths. See the prompt A/B plan for the required measurements and decision gate.

## Repository layout

```text
ARC-AGI-2/
├── configs/
│   ├── qwen3_8b_grpo_lora.yaml
│   └── qwen3_8b_grpo_lora_smoke.yaml
├── docs/
│   ├── IMPLEMENTATION_PLAN.md
│   ├── PROMPT_AB_TEST_PLAN.md
│   └── REASONING_REFINEMENT.md
├── src/arcagi2/
│   ├── config.py
│   ├── data.py
│   ├── episodes.py
│   ├── inspect_data.py
│   ├── metrics.py
│   ├── parsing.py
│   ├── preflight.py
│   ├── prompt_assets/
│   │   ├── student_universal_method.md
│   │   └── student_universal_method_v2.md
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

Optional rollout acceleration and logging:

```bash
pip install -e '.[vllm,wandb]'
```

## Audit before training

The dataset checker reports split statistics, train/evaluation leakage checks, local-test duplication, and augmentation weighting:

```bash
arc-stats --data /path/to/arc-prize-2026-arc-agi-2.zip
```

Then run the model/runtime preflight. It loads the Qwen tokenizer/config but not the 8B weights, checks the installed TRL API, verifies reward dominance, reports the selected prompt method, measures the longest ARC prompts, checks prompt + completion context fit, and reports expected rollout counts:

```bash
arc-preflight \
  --config configs/qwen3_8b_grpo_lora.yaml \
  --data /path/to/arc-prize-2026-arc-agi-2.zip
```

For the supplied competition ZIP, the audited experiment contains **15,350** target×shot training episode specifications per logical cycle and **523** fixed validation query×shot cases.

Run the lightweight tests before launching Qwen:

```bash
pytest -q
```

## Train

Single/multi-GPU launch is handled by Accelerate:

```bash
accelerate launch -m arcagi2.train \
  --config configs/qwen3_8b_grpo_lora.yaml \
  --data /path/to/arc-prize-2026-arc-agi-2.zip
```

The config exposes prompt method, augmentation curriculum, LoRA, GRPO group size, reward weights, optimizer, generation, checkpointing, and optional vLLM settings. The episode stream controls its own logical-cycle order, so TRL's extra iterable-dataset shuffle is disabled.

The default reward weights are:

```text
exact grid      1.00
progress        0.20
answer format   0.02
```

The maximum auxiliary contribution is therefore lower than the exact reward. An incorrect answer cannot outrank an exact answer because of shaping alone. Set the two auxiliary weights to zero for the exact-only ablation.

GRPO logs `arc_exact_grid`, `arc_parseable`, `arc_progress`, `arc_shape_match`, `arc_answer_format`, `arc_group_all_wrong`, `arc_group_all_correct`, and `arc_group_mixed`. `arc_group_mixed` shows how often sampled CoTs provide direct exact-reward contrast; the progress reward is intended to help when exact groups are mostly all-wrong.

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

The validator writes the selected prompt method, overall shot-efficiency score, per-shot accuracies, full-shot two-attempt accuracy, and every evaluated case.
