# V1/V2 prompt smoke evidence

This directory records the V1/V2 prompt regression and the associated exploratory frozen training-only probe captured on 2026-09-12. All cases come from the official training split, with seed 42. The JSON files retain the raw completions and per-case metadata; the `summary` object in each result file contains the aggregate measurements.

## Controlled regression

The regression manifest is `regression_manifest_0d3d703e.json` and reproduces the earlier one-task probe with eight generations per prompt. Both prompt versions reached the exact grid once in eight generations, and both truncated seven of eight generations at the 4,096-token completion cap. V1 took 223.2 seconds total (27.90 seconds/generation); V2 took 214.2 seconds total (26.78 seconds/generation). Each condition had one reward-variation group.

## Exploratory frozen probe

The frozen manifest is `frozen_multitask_manifest.json`. Its V1 file covers 8 prompts and 64 generations: 14 exact/parseable/reached-answer generations, 50 cap truncations, 5 all-zero reward groups, and 3 reward-variation groups. The V2 file contains 1 prompt and 8 generations: all 8 reached a parseable exact answer without truncation, but no reward-variation group was observed. Because the prompt counts differ, these files are evidence of behavior under the recorded runs, not a balanced V1/V2 comparison.

## Environment and preflight snapshot

The matching preflight snapshot is `../20260912_143000_luna/preflight.json`. It records Torch 2.11.0+cu128, Transformers 5.17.0, Datasets 5.0.1, TRL 1.13.0, PEFT 0.20.0, and Accelerate 1.5.1; the local model path was `/raid1/xwan0900/models/Qwen3-8B`, with thinking enabled and a 40,960-token context. The configured training cap was 4,096 completion tokens with truncation masking enabled. Preflight reported 15,350 logical training episode specifications, 122,800 rollouts per logical cycle, 523 validation cases, and a 18,987-token context margin.

## Configuration and repository context

The run used `configs/qwen3_8b_grpo_lora.yaml` with `prompt.method: v2`, one high-to-low-evidence curriculum cycle, eight generations, temperature 0.6, top-p 0.95, top-k 20, and reward weights exact 1.00, progress 0.20, and format 0.02. The smoke config selects the same prompt and generation settings with a five-step training override. The repository README and `docs/PROMPT_AB_TEST_PLAN.md` define the prompt-selection and termination-gate protocol; this archive supplies the measured outputs for that gate.

The raw files are intentionally limited to the two prompt versions, their summaries, and the two manifests. Datasets, model weights, virtual environments, secrets, and bulky training telemetry are excluded.
