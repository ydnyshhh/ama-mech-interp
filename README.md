# AMA Mech Interp

Mechanistic interpretability workspace for the first analysis phase of the agentic moral alignment project.

This repo is scoped to a clean, workshop-oriented first pass:

- Qwen only for the first phase
- sparse checkpoint ladders instead of exhaustive checkpoint sweeps
- behaviorally grounded prompts instead of neuron-first analysis
- representations, probes, and logit-depth before causal tracing

## Repo Layout

The source tree now mirrors the intended analysis stack directly:

- `src/ama_mech_interp/data/gt_harmbench.py`
- `src/ama_mech_interp/data/prompt_suite.py`
- `src/ama_mech_interp/models/load_qwen_adapter.py`
- `src/ama_mech_interp/models/checkpoint_selection.py`
- `src/ama_mech_interp/eval/run_behavior.py`
- `src/ama_mech_interp/extract/save_activations.py`
- `src/ama_mech_interp/analysis/behavior_summary.py`
- `src/ama_mech_interp/analysis/drift.py`
- `src/ama_mech_interp/analysis/probes.py`
- `src/ama_mech_interp/analysis/logit_depth.py`
- `src/ama_mech_interp/analysis/disagreement.py`
- `src/ama_mech_interp/workflow.py`

The pipeline order is encoded directly in code:

1. load checkpoints
2. build prompt suite
3. run behavior
4. extract activations
5. analyze drift, probes, logit depth, and disagreement

## Canonical Run Spec

Every artifact is keyed by the same canonical unit of analysis:

- `model_family`
- `objective`
- `reasoning_mode`
- `tool_mode`
- `checkpoint`
- `prompt_suite`
- `run_id`

The code turns that into a stable run key and uses it in output paths.

## Minimum Viable First Result

The first concrete milestone is:

Compare base, deontological, utilitarian, and game-theoretic Qwen final checkpoints on a 40-prompt subset; save final-position residual-stream requests; produce one behavior table, one layerwise drift job, one action-probe job, and one prompt-level logit-depth job.

This repo now prepares that bundle directly with one command.

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

The preprocessing pipeline now starts from the full cleaned GT-HarmBench master pool and then materializes a smaller controlled suite for phase-one mech-interp work.

The default controlled suite is:

1. `140` analysis prompts
   Balanced across the available GT-HarmBench formal games.
   These are split into `70` probe-train, `35` probe-validation, and `35` probe-test prompts.
2. `30` disagreement prompts
   Proxy-curated before behavior runs using target disagreement and strategic ambiguity.
   This is the initial causal-analysis pool and should later be refreshed from real inter-model disagreement.
3. `30` held-out replication prompts
   Held back for later confirmation once you think you have found a real phenomenon.

The repo materializes these stable artifacts instead of rebuilding them implicitly:

- `artifacts/prompt_pool_master_v1.jsonl`
- `artifacts/prompt_pool_master_v1.csv`
- `artifacts/prompt_suite_v1.jsonl`
- `artifacts/prompt_suite_v1.csv`
- `artifacts/prompt_suite_mvr1.jsonl`
- `artifacts/prompt_suite_manifest.json`
- `artifacts/prompt_suite_v1_build_report.json`

Each prompt row follows a stable schema:

- `prompt_id`
- `source`
- `subset`
- `probe_split`
- `formal_game`
- `formal_game_raw`
- `prompt_text`
- `actions`
- `num_actions`
- `actions_raw`
- `payoff_matrix`
- `payoff_matrix_raw`
- `risk_level`
- `risk_bucket`
- `story_side`
- `payoff_table_present`
- `history_type`
- `target_nash`
- `target_util`
- `target_rawls`
- `target_nsw`
- `target_nash_raw`
- `target_util_raw`
- `target_rawls_raw`
- `target_nsw_raw`

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

### Dataset Pipeline

The canonical dataset build step is now a first-class command.

Build the cleaned master pool and controlled prompt suite:

```powershell
uv run ama-build-prompt-suite --csv gt-harmbench-dataset/gt-harmbench-with-targets.csv --artifacts-root artifacts
```

Equivalent through the main package CLI:

```powershell
uv run python -m ama_mech_interp materialize-prompt-suite --csv gt-harmbench-dataset/gt-harmbench-with-targets.csv --artifacts-root artifacts
```

This writes:

- `artifacts/prompt_pool_master_v1.jsonl`
- `artifacts/prompt_pool_master_v1.csv`
- `artifacts/prompt_suite_v1.jsonl`
- `artifacts/prompt_suite_v1.csv`
- `artifacts/prompt_suite_v1_build_report.json`

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

Generated outputs are organized by pipeline stage:

- `outputs/behavior/...`
- `outputs/activations/...`
- `outputs/drift/...`
- `outputs/probes/...`
- `outputs/logit_depth/...`
- `outputs/disagreement/...`

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
uv run python -m ama_mech_interp dataset-summary
```

Materialize the canonical prompt suite artifacts:

```powershell
uv run python -m ama_mech_interp materialize-prompt-suite
```

Or use the dedicated dataset-build command:

```powershell
uv run ama-build-prompt-suite
```

Prepare the minimum viable end-to-end bundle:

```powershell
uv run python -m ama_mech_interp minimum-viable-pipeline
```

Select a sparse checkpoint ladder from a candidate file:

```powershell
uv run python -m ama_mech_interp select-checkpoints --input checkpoint_candidates.jsonl --output artifacts/checkpoint_selection.json
```

Write a behavior summary after you import observed behavior rows:

```powershell
uv run python -m ama_mech_interp behavior-summary --prompt-suite artifacts/prompt_suite_mvr1.jsonl --behavior-rows outputs/behavior/qwen35_9b__base__native__none__base__prompt_suite_mvr1__mvr1/behavior_rows.jsonl --output outputs/behavior/behavior_summary.json
```

Rank disagreement prompts after you have observed behavior rows from multiple models:

```powershell
uv run python -m ama_mech_interp disagreement --prompt-suite artifacts/prompt_suite_mvr1.jsonl --behavior-rows outputs/behavior/.../behavior_rows.jsonl outputs/behavior/.../behavior_rows.jsonl --output-dir outputs/disagreement/prompt_suite_mvr1__mvr1
```

Print the Qwen model registry or the phase-one analysis plan:

```powershell
uv run python -m ama_mech_interp model-registry
uv run python -m ama_mech_interp phase-one-plan
```
