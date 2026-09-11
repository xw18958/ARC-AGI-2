# Implementation Plan

## 1. Experimental contract

The repository intentionally separates the three roles in the supplied ARC Prize package:

- **Training:** the 1,000 `arc-agi_training_*` tasks.
- **Validation:** the 120 `arc-agi_evaluation_*` tasks. Their solutions are used only for scoring; no gradient update uses them.
- **Local `test_challenges.json`:** not used for training or validation. Kaggle supplies hidden test tasks when the notebook is rerun for scoring.

The model is **Qwen3-8B**, adapted with **LoRA** and **GRPO**. Qwen generates its own reasoning; no labelled chain-of-thought dataset is required. GRPO receives a binary reward of 1 only when the parsed final grid exactly matches the target grid.

## 2. Training augmentation

For every training task, combine its labelled demonstrations with its labelled original test pair(s). If the task contains `M` known pairs:

1. Every pair becomes the target/query once.
2. For each target, cover every shot count `k = 1, ..., M-1`.
3. At access time, sample `k` supports from the other `M-1` pairs and randomly shuffle their order.
4. On the next logical cycle, re-sample the supports/order.

Therefore one logical augmentation cycle contains exactly `M(M-1)` episode specifications per task while avoiding materializing every support subset/permutation. This is analogous to image augmentation: the underlying task is stored once and transformed when used.

The training stream is infinite and cycles through all specifications. TRL receives a bounded `max_steps`, derived from the requested number of logical augmentation epochs. The default is one complete logical cycle; an explicit `max_steps_override` exists only for smoke tests.

## 3. Prompt/answer contract

Each episode contains:

- a system instruction defining ARC grid rules and strict output format;
- `k` solved input-output demonstrations;
- one query grid whose answer is hidden from the model.

Qwen3 thinking mode remains enabled. The model may emit `<think>...</think>`, but its final grid must be in:

```text
<answer>[[...], [...]]</answer>
```

The parser accepts only rectangular grids up to 30x30 containing integer values 0-9.

## 4. GRPO + LoRA

TRL's `GRPOTrainer` is used rather than a custom reinforcement-learning loop.

- LoRA keeps the base Qwen3-8B weights frozen.
- Each prompt produces a group of sampled reasoning completions.
- Reward is binary exact-grid correctness.
- The default config uses TRL's current `dapo` loss normalization to avoid the response-length bias of the original GRPO loss.
- Qwen3's recommended thinking-mode sampling defaults (`temperature=0.6`, `top_p=0.95`, `top_k=20`) are used as initial values.
- vLLM rollout generation can be enabled in config without changing data or reward code.

No partial correctness reward is enabled in the baseline. This keeps the first experiment interpretable.

## 5. Validation protocol

Validation is deterministic in its **task construction**. For each of the 120 evaluation tasks, retain the original demonstration order and original test query(s):

```text
1-shot: A         -> original test query
2-shot: A + B     -> original test query
3-shot: A + B + C -> original test query
...
```

For each condition the inference pipeline samples several completions, parses their final grids, groups identical grids, and selects the two most frequently generated **distinct** grids as the two competition attempts.

### Shot-efficiency score

For a task and shot count, first average correctness across that task's original test queries. Then average across all available shot counts for that task. Finally macro-average the resulting task scores across the 120 tasks.

This means a task with 2 available shots and a task with 6 available shots each contribute exactly one equally weighted task score.

The validator also reports:

- per-shot task-macro accuracy;
- full-original-shot two-attempt accuracy (competition-oriented diagnostic);
- per-case attempts for debugging.

## 6. Scalability

The pipeline is split into independent modules so scaling does not change experimental semantics:

- `data.py`: ZIP/directory I/O and validation.
- `episodes.py`: dynamic augmentation and fixed validation cases.
- `prompts.py`: one prompt contract shared by training/validation.
- `parsing.py`: one strict answer parser shared by reward/validation.
- `rewards.py`: stateless exact-match reward for distributed GRPO.
- `metrics.py`: aggregation independent of model execution.
- `train.py`: TRL/PEFT orchestration; compatible with Accelerate and optional vLLM.
- `validate.py`: batched inference; candidate voting is independent of batch size.

Large generated artifacts, datasets, checkpoints and W&B runs are ignored by Git.

## 7. Execution order

1. Install the package: `pip install -e '.[dev]'` (add `vllm` or `wandb` extras if needed).
2. Verify the dataset: `arc-stats --data /path/to/arc-prize-2026-arc-agi-2.zip`.
3. Run unit tests: `pytest -q`.
4. Smoke-test base Qwen validation on a small case subset if desired.
5. Launch GRPO: `accelerate launch -m arcagi2.train --config configs/qwen3_8b_grpo_lora.yaml --data ...`.
6. Validate a saved adapter: `arc-validate --data ... --checkpoint outputs/.../checkpoint-N`.
7. Select checkpoints by **shot-efficiency score**; keep full-shot two-attempt accuracy as the competition-oriented secondary metric.
