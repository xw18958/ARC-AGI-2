# Prompt V1 vs V2 A/B Test Plan

## Purpose

The first GPU smoke test on task `0d3d703e` showed a pre-answer failure: all eight Qwen3-8B generations exhausted the 4,096-token completion budget inside `<think>`, never closed `</think>`, never reached `<answer>`, and therefore produced zero trainable GRPO reward.

The immediate experiment must test whether the refined prompt improves reasoning termination and task solving **without changing training, reward, sampling, or token-budget settings at the same time**.

## Prompt variants

- `v1`: `src/arcagi2/prompt_assets/student_universal_method.md` — original empirical method guide derived from prior GPT-based ARC experiments.
- `v2`: `src/arcagi2/prompt_assets/student_universal_method_v2.md` — compact controller plus the same specialized method families in compressed trigger-based form.

Select with:

```yaml
prompt:
  method: v1  # or v2
```

The main and smoke configs select `v2`. For A/B testing, make a copied config that differs only in `prompt.method`.

## What V2 is intended to fix

1. ARC color IDs are explicitly treated as categorical symbols rather than numerical quantities.
2. Simple symbol substitution is tested early when geometry is unchanged.
3. Specialized methods are consulted only when their trigger matches visible evidence.
4. Rejected hypotheses should not be repeatedly reconsidered.
5. A verified rule is an explicit stop condition: once it reproduces all demonstrations, stop searching and answer.
6. All empirically discovered specialized method families from V1 are retained, but compressed.

## Experimental controls

For every V1/V2 comparison keep identical:

- exact model checkpoint;
- task/query/support identities and support order;
- seed;
- `enable_thinking`;
- temperature, top-p, top-k, min-p;
- number of generations;
- maximum completion length;
- tokenizer/chat template;
- GPU/runtime stack;
- reward functions;
- answer parser.

Do **not** simultaneously change `max_completion_length`, presence penalty, reward weights, curriculum, LoRA, or GRPO settings. A prompt comparison is interpretable only if prompt selection is the intervention.

## Stage 1 — regression task

Re-run the exact failed probe for task `0d3d703e` with the same supports, seed, sampling, eight generations, and 4,096-token cap under V1 and V2.

For each generation record:

- completion token count;
- EOS reached;
- `</think>` reached;
- `<answer>` reached;
- parseable final grid;
- exact correctness;
- progress reward;
- format reward;
- total weighted reward;
- generation time;
- whether the reasoning repeatedly revisits the same rejected hypothesis.

The V2 trace should be inspected specifically for the expected representation shift:

```text
same dimensions / same geometry
-> only symbols change
-> infer symbolic color mapping
-> verify mapping on all demonstrations
-> stop
-> answer
```

Do not declare V2 successful only because it produces shorter text. The main outcomes are answer-reached rate and correctness.

## Stage 2 — multi-task controlled probe

One task is insufficient. Build a fixed probe set using **training tasks only**; do not use the 120 official evaluation tasks for prompt development.

The probe set should deliberately include different structures and shot counts rather than only easy or small grids. Freeze the selected task IDs, target indices, support identities, support order, and random seeds in an artifact committed to the repo so later reruns use the exact same cases.

Run both V1 and V2 on exactly those frozen cases.

Aggregate at minimum:

- `% reached </think>`;
- `% reached <answer>`;
- `% parseable answer`;
- exact-grid accuracy;
- median terminated completion length;
- p90/p95 terminated completion length when enough terminated samples exist;
- 4,096-token truncation rate;
- reward-zero-group rate;
- wall-clock generation time;
- obvious repetition-loop rate using a deterministic, documented detector or manual audit subset.

Also report results by task/shot condition so improvements are not hidden by aggregation.

## Stage 3 — completion-length diagnosis

Do not choose a new training completion cap by guessing a round number.

If V2 still has substantial truncation at 4,096, run a separate **diagnostic** generation study to observe natural termination lengths. Use a ceiling that is supported by the model/runtime context and clearly label it as a diagnostic ceiling, not a proposed training cap.

From naturally terminated samples, report the empirical completion-length distribution. Only after observing that distribution should a future `max_completion_length` be proposed. Any proposed cap must be justified by the measured coverage/truncation trade-off.

If trajectories continue looping far beyond 4,096 rather than converging, treat that as a reasoning-loop problem rather than evidence that the training cap simply needs to be larger.

## Stage 4 — decision gate before GRPO

Do not start the five-step GRPO smoke test until V2 produces a meaningful number of completed, parseable answers on the controlled probe.

Proceed to GRPO only if:

1. the prompt path is verified end-to-end;
2. V2 reaches answers materially more often than the failed baseline or otherwise demonstrates acceptable termination on the frozen probe;
3. reward groups can contain non-identical rewards rather than all eight generations being truncated zeros;
4. no train/evaluation leakage has been introduced.

If V2 reaches answers but nearly every answer is wrong, consider a separate self-generated successful-trace SFT warm-up experiment before GRPO. That is a new intervention and must be evaluated separately.

## Required artifacts

Store a new timestamped directory under `outputs/smoke_evidence/` containing:

- exact git commit SHA;
- environment/package versions;
- GPU information;
- frozen probe manifest;
- V1 raw generations and summary;
- V2 raw generations and summary;
- per-generation termination/reward fields;
- aggregate comparison table;
- representative traces showing success/failure modes;
- commands/configs used;
- a short conclusion with one of:
  - `V2 PROMPT READY FOR GRPO SMOKE`
  - `V2 IMPROVES TERMINATION BUT NEEDS MORE DIAGNOSIS`
  - `V2 DOES NOT IMPROVE THE FAILURE MODE`
  - `IMPLEMENTATION/EXPERIMENT INVALID`

Do not launch full training as part of this prompt experiment.