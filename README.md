# AMA Mech Interp

Mechanistic interpretability workspace for the first analysis phase of the agentic moral alignment project.

This repo is scoped to a clean, workshop-oriented first pass:

- Qwen only for the first phase
- sparse checkpoint ladders instead of exhaustive checkpoint sweeps
- behaviorally grounded prompts instead of neuron-first analysis
- representations, probes, and logit-depth before causal tracing

## Phase One Goal

Support this core question:

Do deontological, utilitarian, and game-theoretic fine-tuning objectives induce distinct internal moral decision policies, or do they mostly preserve a shared strategic representation and differ only in late-stage readout or decision layers?

## Models

Use only the Qwen family in phase one:

- `unsloth/Qwen3.5-9B`
- `agentic-moral-alignment/qwen35-9b__ipd_str_tft__deont__native_tool__r1`
- `agentic-moral-alignment/qwen35-9b__ipd_str_tft__deont__native_notool__r20`
- `agentic-moral-alignment/qwen35-9b__ipd_str_tft__util__native_tool__r1`
- `agentic-moral-alignment/qwen35-9b__ipd_str_tft__game__native_tool__r1`

The Hugging Face organization also exposes `agentic-moral-alignment/checkpoints` and `agentic-moral-alignment/runs`, which should be used to pick sparse, behavior-anchored checkpoint ladders when available.

## Recommended Checkpoint Policy

For each finetuned model, select five checkpoints:

1. Before the first clear behavioral shift
2. First clear behavioral shift
3. Middle training
4. Late training
5. Final checkpoint

If eval logs are incomplete, fall back to an evenly spread ladder such as early, early-mid, mid, late, and final. Do not compare arbitrary step numbers across objectives when the behaviors shift at different rates.

## Prompt Suite

Build an approximately 200-prompt workshop suite with three blocks:

1. `80` repeated-game prompts
   Focus on IPD, iterated coordination, Chicken, and Stag variants.
   Include history conditions: no history, cooperation streak, betrayal streak, alternating, tit-for-tat-like, and random-like.
2. `80` one-shot GT-HarmBench prompts
   Use the local CSV as the master source.
   Prioritize Prisoner's Dilemma, Stag Hunt, Chicken, Bach or Stravinski, and Coordination.
   Keep `story_row` only in phase one.
3. `40` disagreement prompts
   Reserve for prompts where target labels disagree or the models later disagree behaviorally.
   These become the patching shortlist.

### GT-HarmBench Guidance

Use the local dataset at `gt-harmbench-dataset/gt-harmbench-with-targets.csv` as the master source for one-shot prompts.

Most useful columns:

- `id`
- `formal_game`
- `story_row`
- `actions_row`
- `1_1_payoff`, `1_2_payoff`, `2_1_payoff`, `2_2_payoff`
- `risk_level`
- `quality_score`
- `equilibria_score`
- `target_nash_equilibria`
- `target_utility_maximizing`
- `target_rawlsian`
- `target_nash_social_welfare`

First cleaning pass:

- drop rows with missing `story_row`
- normalize `formal_game`
- parse action lists
- parse payoff cells
- normalize target labels

## Analyses To Run First

Run these in priority order.

### 1. Behavioral Summary

Goal:
Find where the models actually differ before interpreting activations.

Generate:

- action distributions by model and prompt family
- pairwise disagreement matrices across objectives
- a ranked table of prompts with the clearest inter-model disagreement

### 2. Layerwise Representation Drift

Goal:
Find where fine-tuning changes hidden-state geometry, and whether those changes follow shared or objective-specific depth profiles.

Generate:

- base-to-checkpoint CKA curves by layer
- inter-objective CKA curves by layer
- checkpoint trajectory heatmaps
- cosine or Procrustes drift plots if you want a second similarity measure

### 3. Linear Probe Sweeps

Goal:
Measure when relevant information becomes linearly accessible across depth.

Probe targets:

- chosen action
- cooperate versus defect style action class
- true game family
- history condition for repeated-game prompts
- betrayal or cooperation state where available

Generate:

- probe accuracy by layer and checkpoint for each target
- cross-objective transfer probe results to test shared versus objective-specific structure

### 4. Logit Through Depth

Goal:
Track when action commitment emerges.

Generate:

- cooperate-versus-defect or analogous action margin by layer
- prompt-level depth traces for top disagreement prompts
- checkpoint comparison panels for early, mid, and final models

### 5. Disagreement Mining

Goal:
Produce a compact patching shortlist for later causal tracing.

Rank prompts by:

- inter-model disagreement
- target-label disagreement
- action margin confidence
- parsing cleanliness
- family and risk diversity

## Strong Evidence Patterns

### Shared Strategic Representation With Late Objective Readout

Treat this as strongly supported if you see most of the following:

- early and middle layers remain highly similar across objectives
- game-family and state probes transfer well across objectives in early and middle depth
- divergence is concentrated in late-layer action probes or logit-difference curves
- disagreement prompts separate objectives mainly near the decision readout

### Deeper Objective-Specific Reorganization

Treat this as strongly supported if you see most of the following:

- CKA or related similarity drops already in early or middle layers
- probe transfer between objectives degrades before the final third of the network
- disagreement prompts separate objective families in hidden-state space well before action readout
- checkpoint trajectories show different depth-localized change profiles for deontological, utilitarian, and game-theoretic tuning

### State-Dependent Moral Computation

Treat this as strongly supported if you see most of the following:

- repeated-game history changes action margins through depth within the same model
- those state effects differ systematically across objectives
- history-conditioned probes become accessible before the final layer
- the same one-shot prompt looks different once you add betrayal or cooperation history

## Save These Tensors

Save:

- residual stream at the decision position for every layer
- final-token hidden state by layer
- chosen action logits and action margins
- prompt metadata for every run

Optionally save:

- selected attention patterns for top disagreement prompts and selected layers
- selected MLP outputs for top disagreement prompts and selected layers

Do not save in phase one:

- full attention weights for every layer and prompt
- exhaustive head-level traces
- neuron-level activation dumps for everything

## Output Layout

Use flat, analysis-friendly tables.

### `prompt_table`

One row per prompt variant.

Recommended columns:

- `prompt_id`
- `prompt_source`
- `variant_name`
- `game_family`
- `history_type`
- `payoff_table_present`
- `risk_level`
- `target_nash_equilibria`
- `target_utility_maximizing`
- `target_rawlsian`
- `target_nash_social_welfare`

### `run_table`

One row per `model x checkpoint x prompt`.

Recommended columns:

- `model_key`
- `objective`
- `inference_condition`
- `checkpoint_label`
- `prompt_id`
- `chosen_action`
- `action_margin`
- `reasoning_trace_available`

### `activation_index`

One row per saved tensor artifact.

Recommended columns:

- `model_key`
- `checkpoint_label`
- `prompt_id`
- `layer_index`
- `tensor_name`
- `storage_path`

### `probe_result_table`

One row per `probe_target x model x checkpoint x layer`.

Recommended columns:

- `probe_target`
- `model_key`
- `checkpoint_label`
- `layer_index`
- `metric_name`
- `metric_value`

## Failure Modes And Confounds

- checkpoint selection bias: behavior-anchored selection is better than raw step spacing
- prompt imbalance: stratify game family, risk, and target disagreement
- action-token mismatch: canonicalize action labels before scoring
- probe overinterpretation: a successful linear probe is not a localized causal mechanism
- adapter-format mismatch: normalize the loading and hook points before comparing activations
- reasoning-trace leakage: measure at a stable decision position, not just wherever the model becomes verbose

## Local Commands

Use `uv` from this repo.

Print the GT-HarmBench summary:

```powershell
uv run python -m ama_mech_interp.cli dataset-summary
```

Print the phase-one plan:

```powershell
uv run python -m ama_mech_interp.cli phase-one-plan
```

Print the Qwen model registry:

```powershell
uv run python -m ama_mech_interp.cli model-registry
```
