# Reasoning-Training Refinement

## Goal

Train Qwen3-8B to discover better ARC reasoning trajectories without a teacher chain-of-thought dataset. Qwen generates its own `<think>...</think>` trajectory; GRPO assigns an advantage to the **whole generated sequence** from verifiable properties of the final grid.

The design intentionally does **not** reward reasoning length, particular phrases, or a hand-written explanation. Those signals are easy to game and are not guaranteed to correspond to correct ARC reasoning.

## Reward design

Three verifiable reward functions are combined:

1. **Exact grid reward — weight 1.00.** `1` only when the predicted grid exactly equals the ground truth, otherwise `0`. This is the real ARC objective and remains dominant.
2. **Grid progress reward — weight 0.20.** Invalid or wrong-shape outputs receive `0`. A same-shape grid receives a small shape floor plus exact cell-level accuracy. This creates relative signal when all sampled trajectories miss the full answer.
3. **Answer-format reward — weight 0.02.** Rewards a parseable ARC grid inside an explicit `<answer>...</answer>` block. This reduces wasted rollouts from malformed outputs.

The maximum auxiliary contribution is `0.22`, which is below the `1.00` exact reward. Therefore an incorrect trajectory can never outrank an exact trajectory because of shaping alone.

For the exact-only ablation, set:

```yaml
rewards:
  exact_weight: 1.0
  progress_weight: 0.0
  format_weight: 0.0
```

## Why this can improve CoT

GRPO samples several complete reasoning trajectories for the same prompt. Rewards are computed from their final answers, then the policy update changes the probability of **all generated tokens in those trajectories**, including the reasoning tokens. A trajectory leading to a better verified output therefore receives a better advantage even though no labelled CoT is supplied.

The dense progress reward is only a bootstrap signal. Exact output correctness remains the optimization target.

## Evidence curriculum

Every training task still covers every target pair and every valid shot count exactly once per logical cycle. During the initial configured cycle(s), episode specifications are ordered from **higher evidence to lower evidence** using:

`shot_count / maximum_available_shots`

Supports and their order are still randomly sampled. After the curriculum cycles, the specifications are fully shuffled.

This does **not** assume that ARC tasks with more demonstrations are intrinsically easier. It only exploits the fact that, within the same task, a higher-shot condition exposes more evidence. Validation is unaffected and remains fixed.

## Stability choices

- `scale_rewards: false` avoids standard-deviation scaling that can introduce question-difficulty reweighting.
- `loss_type: dapo` avoids the response-length bias of the original GRPO token averaging.
- `mask_truncated_completions: true` prevents cut-off reasoning trajectories that never finish an answer from producing noisy policy updates.
- Qwen thinking-mode sampling starts at `temperature=0.6`, `top_p=0.95`, `top_k=20`.

## Diagnostics to inspect during the smoke run

The reward functions log:

- `arc_exact_grid`: exact-answer rate.
- `arc_parseable`: parseable output rate.
- `arc_progress`: average progress reward.
- `arc_shape_match`: correct-output-shape rate.
- `arc_answer_format`: strict answer-format rate.
- `arc_group_all_wrong`: groups with no exact solution.
- `arc_group_all_correct`: groups with all exact solutions.
- `arc_group_mixed`: groups containing both correct and incorrect trajectories.

`arc_group_mixed` is especially useful: it tells us how often the exact reward alone gives relative signal. The progress reward is most valuable when exact groups are predominantly all-wrong.

## Refinement decision rule

Start with the configured shaped reward. Keep the exact-only configuration as an ablation. If progress reward increases training reward but does not improve the fixed validation shot-efficiency score, remove or reduce it rather than increasing its weight.
