from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class Hypothesis:
    name: str
    prediction: str
    strong_evidence: str


@dataclass(frozen=True)
class AnalysisSpec:
    name: str
    priority: int
    purpose: str
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class PromptGroup:
    name: str
    target_count: int
    source: str
    rationale: str


@dataclass(frozen=True)
class TensorSpec:
    name: str
    save_policy: str
    rationale: str


@dataclass(frozen=True)
class DataTableSpec:
    name: str
    grain: str
    key_columns: tuple[str, ...]


@dataclass(frozen=True)
class PhaseOnePlan:
    research_question: str
    minimum_viable_result: str
    hypotheses: tuple[Hypothesis, ...]
    analyses: tuple[AnalysisSpec, ...]
    prompt_groups: tuple[PromptGroup, ...]
    tensors: tuple[TensorSpec, ...]
    data_tables: tuple[DataTableSpec, ...]
    confounds: tuple[str, ...]


def buildPhaseOnePlan() -> PhaseOnePlan:
    return PhaseOnePlan(
        research_question=(
            "Do deontological, utilitarian, and game-theoretic fine-tuning objectives induce "
            "distinct internal moral decision policies, or do they mostly preserve a shared "
            "strategic representation and differ only in late-stage readout or decision layers?"
        ),
        minimum_viable_result=(
            "Compare base, deontological, utilitarian, and game-theoretic Qwen final checkpoints "
            "on a 40-prompt subset; prepare behavior rows, activation requests, and drift/probe/logit-depth jobs."
        ),
        hypotheses=(
            Hypothesis(
                name="shared_strategy_late_readout",
                prediction=(
                    "Game-state and payoff-structure information should stay aligned across objectives "
                    "through early and middle layers, while decision-aligned logits and action probes diverge "
                    "mainly in late residual stream layers."
                ),
                strong_evidence=(
                    "Low inter-objective drift through early and mid depth, strong cross-objective game-family "
                    "probe transfer, and divergence concentrated in late action probes or logit-difference curves."
                ),
            ),
            Hypothesis(
                name="objective_specific_reorganization",
                prediction=(
                    "Different alignment objectives should alter representational geometry earlier in the network, "
                    "especially on prompts where Nash, utilitarian, and Rawlsian targets disagree."
                ),
                strong_evidence=(
                    "Early or middle-layer CKA collapse between objectives, probe feature transfer degrading well "
                    "before the decision layers, and disagreement prompts separating objective families in representation space."
                ),
            ),
            Hypothesis(
                name="state_dependent_moral_computation",
                prediction=(
                    "Repeated-game histories such as cooperation, betrayal, and tit-for-tat should modulate the same "
                    "decision pathway differently across objectives, rather than only changing output text style."
                ),
                strong_evidence=(
                    "History-conditioned logit trajectories and action probes shift in a structured way within the same model, "
                    "with the effect appearing before the final readout position."
                ),
            ),
        ),
        analyses=(
            AnalysisSpec(
                name="behavioral_summary",
                priority=1,
                purpose=(
                    "Establish where the models actually differ before interpreting activations. "
                    "This defines the later causal-analysis prompt set."
                ),
                outputs=(
                    "action_distribution_by_model_and_prompt_family",
                    "pairwise_disagreement_matrix",
                    "high_value_prompt_table",
                ),
            ),
            AnalysisSpec(
                name="layerwise_representation_drift",
                priority=1,
                purpose=(
                    "Locate which layers move during fine-tuning and whether objectives share or split the same depth profile."
                ),
                outputs=(
                    "base_to_checkpoint_cka_curve",
                    "inter_objective_cka_curve",
                    "checkpoint_trajectory_heatmap",
                ),
            ),
            AnalysisSpec(
                name="linear_probe_sweep",
                priority=2,
                purpose=(
                    "Measure when action, game family, and state variables become linearly available across depth and training."
                ),
                outputs=(
                    "action_probe_accuracy_by_layer",
                    "game_family_probe_accuracy_by_layer",
                    "state_probe_accuracy_by_layer",
                ),
            ),
            AnalysisSpec(
                name="logit_through_depth",
                priority=2,
                purpose=(
                    "Track where the model commits to its action choice and whether objectives differ in the timing of commitment."
                ),
                outputs=(
                    "action_margin_curve_by_layer",
                    "prompt_level_logit_depth_panels",
                ),
            ),
            AnalysisSpec(
                name="behavioral_disagreement_mining",
                priority=3,
                purpose=(
                    "Rank prompts for later activation patching by combining behavioral divergence, label disagreement, and clean parsing."
                ),
                outputs=(
                    "candidate_patching_prompt_rankings",
                    "disagreement_prompt_metadata_table",
                ),
            ),
        ),
        prompt_groups=(
            PromptGroup(
                name="repeated_game_core",
                target_count=80,
                source="template_generated",
                rationale=(
                    "Main focus. Build stateful IPD, iterated coordination, Chicken, and Stag prompts with history variants: "
                    "no history, cooperation streak, betrayal streak, alternating, tit-for-tat-like, and random-like."
                ),
            ),
            PromptGroup(
                name="gt_harmbench_one_shot_core",
                target_count=80,
                source="gt_harmbench_with_targets",
                rationale=(
                    "Balanced one-shot controls with explicit payoff structure. Prioritize Prisoner's Dilemma, Stag Hunt, "
                    "Chicken, Bach or Stravinski, and Coordination while keeping row-player narration only."
                ),
            ),
            PromptGroup(
                name="disagreement_priority_set",
                target_count=40,
                source="gt_harmbench_plus_model_behavior",
                rationale=(
                    "Reserve for high-value causal tracing. Fill with prompts where base, deontological, utilitarian, "
                    "and game-theoretic models disagree or where target labels disagree sharply."
                ),
            ),
        ),
        tensors=(
            TensorSpec(
                name="residual_stream_at_decision_position",
                save_policy="save_for_all_layers_all_models_all_selected_prompts",
                rationale="Core tensor for drift, probe sweeps, and later activation patching.",
            ),
            TensorSpec(
                name="final_token_hidden_state_by_layer",
                save_policy="save_for_all_layers_all_models_all_selected_prompts",
                rationale="Cheap summary view for probe baselines and early plotting.",
            ),
            TensorSpec(
                name="action_token_logits_and_logit_margins",
                save_policy="save_for_all_models_all_selected_prompts",
                rationale="Needed for behavioral summaries and logit-through-depth analysis.",
            ),
            TensorSpec(
                name="selected_attention_patterns",
                save_policy="save_only_for_top_disagreement_prompts_and_selected_layers",
                rationale="Useful later, but not first-pass core infrastructure.",
            ),
            TensorSpec(
                name="selected_mlp_outputs",
                save_policy="save_only_for_top_disagreement_prompts_and_selected_layers",
                rationale="Optional for narrowing candidate mechanisms after drift and probe results.",
            ),
            TensorSpec(
                name="full_attention_weights_all_layers",
                save_policy="do_not_save_in_phase_one",
                rationale="High storage cost and low first-pass signal-to-noise.",
            ),
        ),
        data_tables=(
            DataTableSpec(
                name="prompt_table",
                grain="one_row_per_prompt_variant",
                key_columns=(
                    "prompt_id",
                    "prompt_source",
                    "game_family",
                    "variant_name",
                    "history_type",
                    "payoff_table_present",
                    "risk_level",
                    "target_nash_equilibria",
                    "target_utility_maximizing",
                    "target_rawlsian",
                    "target_nash_social_welfare",
                ),
            ),
            DataTableSpec(
                name="run_table",
                grain="one_row_per_model_checkpoint_prompt",
                key_columns=(
                    "model_key",
                    "objective",
                    "inference_condition",
                    "checkpoint_label",
                    "prompt_id",
                    "chosen_action",
                    "action_margin",
                    "reasoning_trace_available",
                ),
            ),
            DataTableSpec(
                name="activation_index",
                grain="one_row_per_saved_tensor",
                key_columns=(
                    "model_key",
                    "checkpoint_label",
                    "prompt_id",
                    "layer_index",
                    "tensor_name",
                    "storage_path",
                ),
            ),
            DataTableSpec(
                name="probe_result_table",
                grain="one_row_per_probe_target_model_checkpoint_layer",
                key_columns=(
                    "probe_target",
                    "model_key",
                    "checkpoint_label",
                    "layer_index",
                    "metric_name",
                    "metric_value",
                ),
            ),
        ),
        confounds=(
            "Checkpoint selection bias: prefer behavior-anchored checkpoints when eval runs are available; do not compare arbitrary step numbers if objectives improve at different speeds.",
            "Prompt imbalance: keep family, risk, and target-disagreement strata explicit so objective effects are not really data-mixture effects.",
            "Action-token mismatch: canonicalize action labels and score by action set membership, not naive first-token overlap.",
            "Probe overinterpretation: use held-out prompts, cross-checkpoints, and cross-objective transfer; a high probe score does not imply a localized causal feature.",
            "Adapter-format mismatch: adapter exports and full-model exports should be normalized before comparing hidden states.",
            "Reasoning-trace leakage: do not let verbose reasoning tokens define the measurement position; compare at a consistent action-decision location.",
        ),
    )


def renderPlanAsMarkdown() -> str:
    plan = buildPhaseOnePlan()
    lines: list[str] = []
    lines.append("# Phase One Analysis")
    lines.append("")
    lines.append("## Research Question")
    lines.append(plan.research_question)
    lines.append("")
    lines.append("## Minimum Viable Result")
    lines.append(plan.minimum_viable_result)
    lines.append("")
    lines.append("## Hypotheses")
    for hypothesis in plan.hypotheses:
        lines.append(f"- {hypothesis.name}: {hypothesis.prediction}")
        lines.append(f"  Strong evidence: {hypothesis.strong_evidence}")
    lines.append("")
    lines.append("## Analyses")
    for analysis in sorted(plan.analyses, key=lambda item: item.priority):
        lines.append(f"- P{analysis.priority} {analysis.name}: {analysis.purpose}")
        lines.append(f"  Outputs: {', '.join(analysis.outputs)}")
    lines.append("")
    lines.append("## Prompt Groups")
    for group in plan.prompt_groups:
        lines.append(f"- {group.name} ({group.target_count}): {group.rationale}")
    lines.append("")
    lines.append("## Save These Tensors")
    for tensor in plan.tensors:
        lines.append(f"- {tensor.name}: {tensor.save_policy}. {tensor.rationale}")
    lines.append("")
    lines.append("## Tables")
    for table in plan.data_tables:
        lines.append(f"- {table.name} [{table.grain}]")
        lines.append(f"  Columns: {', '.join(table.key_columns)}")
    lines.append("")
    lines.append("## Confounds")
    for confound in plan.confounds:
        lines.append(f"- {confound}")
    return "\n".join(lines)


def renderPlanAsJson() -> str:
    plan = buildPhaseOnePlan()
    return json.dumps(asdict(plan), indent=2)
