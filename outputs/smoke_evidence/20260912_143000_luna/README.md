# Luna smoke-test evidence

This directory records the deterministic checks run against the packaged
student universal ARC method prompt on 2026-09-12.

## Result

The implementation and data checks passed, but the controlled generation probe
did not produce a trainable signal at the configured 4,096-token completion
limit. All eight trajectories exhausted the limit inside native reasoning:

- EOS: `0/8`
- closed `</think>`: `0/8`
- explicit `<answer>` block: `0/8`
- parseable grids: `0/8`
- nonzero rewards: `0/8`
- peak VRAM: `21.89 GiB`
- generation time: `218.9 s`

The five-step GRPO run was therefore not started. `train_5step.log` contains
the partial model-loading launch log; no optimizer step or checkpoint is
claimed by this evidence.

## Artifacts

- `controlled_probe_0d3d703e.json`: raw eight-trajectory probe and reward data.
- `preflight.json`: installed versions, context budget, prompt maximum, and
  GRPO API compatibility check.
- `arc_stats.json`: official train/evaluation counts and overlap audit.
- `two_cycle_episode_audit.txt`: full episode coverage and validation-case
  audit for two logical cycles.
- `synthetic_reward_ordering.txt`: deterministic reward ordering checks.
- `train_5step.log`: partial training launch log retained for diagnosis.
