# Implementation Plan

## 1. Experimental contract

The supplied ARC Prize package has three separate roles:

- **Training:** 1,000 `arc-agi_training_*` tasks.
- **Validation:** 120 `arc-agi_evaluation_*` tasks. Their solutions are used only for scoring; no gradient update uses them.
- **Local `test_challenges.json`:** excluded from development because the supplied 240 tasks duplicate training tasks; Kaggle substitutes hidden tasks for scoring.

The model is **Qwen3-8B**, adapted with **LoRA + GRPO**. Qwen generates its own reasoning. No labelled chain-of-thought dataset is required.

## 2. Training augmentation

For every training task, combine its labelled demonstrations with its labelled original test pair(s). If a task contains `M` known pairs:

1. Every pair becomes the target/query once.
2. For each target, cover every shot count `k = 1, ..., M-1`.
3. At access time, randomly sample `k` supports from the remaining pairs and shuffle their order.
4. On later logical cycles, re-sample support identities/order.

One logical cycle therefore contains exactly `M(M-1)` target×shot episode specifications per task without materializing every support subset/permutation.

### Initial evidence curriculum

Coverage never changes. During the configured initial cycle(s), specifications are ordered by the fraction of available demonstrations shown, from high evidence to low evidence. After those cycles the specification order is fully shuffled.

This is **not** an assumption that tasks with more demonstrations are easier. It is only a bootstrap ordering within the variable-shot training distribution.

## 3. Prompt/answer contract

Each episode contains:

- a system instruction defining ARC grid rules and strict output format;
- `k` solved input-output demonstrations;
- one query grid whose output is hidden from the model.

Qwen3 thinking mode remains enabled. The model can generate its native `<think>...</think>` trajectory and must place the final grid in:

```text
<answer>[[...], [...]]</answer>
```

The parser accepts only rectangular grids up to 30×30 with integer values 0–9.

## 4. Training better reasoning with GRPO

GRPO samples multiple complete Qwen reasoning trajectories for each ARC episode. Reward is computed from the final output, but the policy update applies to the generated trajectory, including its reasoning tokens. This lets Qwen improve its own CoT without a teacher rationale.

The default reward is verifiable and hierarchical:

- **Exact grid:** weight `1.00`, binary exact match.
- **Grid progress:** weight `0.20`, zero for invalid/wrong-shape grids; otherwise a shape floor plus cell-level exact accuracy.
- **Strict answer format:** weight `0.02`, requiring a parseable `<answer>` grid.

Exact correctness is guaranteed to dominate all auxiliary rewards combined. The exact-only ablation is available by setting the two auxiliary weights to zero.

We deliberately do not reward CoT length, wording, or unsupported reasoning-process heuristics. See [`REASONING_REFINEMENT.md`](REASONING_REFINEMENT.md).

### GRPO stability settings

- `scale_rewards: false` avoids standard-deviation-based question-difficulty reweighting.
- `loss_type: dapo` avoids the original GRPO response-length normalization bias.
- `mask_truncated_completions: true` excludes cut-off reasoning trajectories from the policy loss.
- Default group size is 8 completions per prompt.
- Qwen thinking-mode sampling begins at `temperature=0.6`, `top_p=0.95`, `top_k=20`.
- Optional vLLM rollout generation changes execution speed, not experiment semantics.

## 5. Validation protocol

Validation is fixed and unaffected by training augmentation/curriculum. For every one of the 120 evaluation tasks, retain the original demonstration order and original test query(s):

```text
1-shot: A         -> original test query
2-shot: A + B     -> original test query
3-shot: A + B + C -> original test query
...
```

For each condition, inference samples multiple completions, parses their final grids, groups identical grids, and selects the two most frequent **distinct** grids as the two competition attempts.

### Shot-efficiency score

For each task:

1. average correctness across its original test queries at each shot level;
2. average those shot-level scores across all available shots;
3. macro-average the resulting task scores over all 120 tasks.

Thus every validation task contributes equal final weight regardless of how many shots/test queries it contains.

Also report:

- per-shot task-macro accuracy;
- full-original-shot two-attempt exact-match accuracy;
- per-case attempts for debugging.

## 6. Scalability and reproducibility

Modules remain independent:

- `data.py`: ZIP/directory loading and split integrity.
- `episodes.py`: dynamic augmentation, curriculum ordering, fixed validation cases.
- `prompts.py`: shared train/validation prompt contract.
- `parsing.py`: strict grid extraction.
- `rewards.py`: exact/progress/format rewards and sparse-reward diagnostics.
- `metrics.py`: validation aggregation independent of inference execution.
- `preflight.py`: tokenizer/context/TRL API check before loading 8B weights.
- `train.py`: TRL/PEFT orchestration, Accelerate, optional vLLM.
- `validate.py`: batched inference and candidate voting.

The episode stream itself controls ordering, so TRL's extra iterable-dataset shuffle is disabled. Large datasets, checkpoints, generated outputs, and W&B runs are ignored by Git.

## 7. Verification sequence

1. `pip install -e '.[dev]'` (plus `vllm`/`wandb` extras if needed).
2. `arc-stats --data DATA.zip` — verify 1,000 train / 120 eval, no train↔eval leakage, and expected episode counts.
3. `pytest -q` — verify augmentation, curriculum coverage, parser, rewards, candidate voting, and metrics.
4. `arc-preflight --config configs/qwen3_8b_grpo_lora.yaml --data DATA.zip` — verify installed TRL fields, tokenizer/context budget, reward config, and expected rollout counts.
5. Run a short GRPO smoke test and inspect `arc_exact_grid`, `arc_progress`, `arc_parseable`, `arc_group_all_wrong`, and `arc_group_mixed`.
6. Only then run a complete logical cycle.
7. Validate saved adapters with the fixed cumulative-shot protocol.
8. Select checkpoints by **shot-efficiency score**; use full-shot two-attempt accuracy as the competition-oriented secondary metric.
