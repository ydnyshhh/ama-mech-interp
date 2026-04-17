from __future__ import annotations

from dataclasses import dataclass


BASE_MODEL_REPO = "unsloth/Qwen3.5-9B"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    architecture_family: str
    objective: str
    inference_condition: str
    repo_id: str
    adapter_format: str
    phase_one_priority: int


@dataclass(frozen=True)
class CheckpointPolicy:
    name: str
    checkpoint_count: int
    target_events: tuple[str, ...]
    selection_rule: str


def getPhaseOneModels() -> list[ModelSpec]:
    return [
        ModelSpec(
            key="qwen_base",
            architecture_family="qwen35_9b",
            objective="base",
            inference_condition="native_base",
            repo_id=BASE_MODEL_REPO,
            adapter_format="base_model",
            phase_one_priority=1,
        ),
        ModelSpec(
            key="qwen_deont_tool",
            architecture_family="qwen35_9b",
            objective="deontological",
            inference_condition="native_tool",
            repo_id="agentic-moral-alignment/qwen35-9b__ipd_str_tft__deont__native_tool__r1",
            adapter_format="peft_adapter",
            phase_one_priority=1,
        ),
        ModelSpec(
            key="qwen_deont_notool",
            architecture_family="qwen35_9b",
            objective="deontological",
            inference_condition="native_notool",
            repo_id="agentic-moral-alignment/qwen35-9b__ipd_str_tft__deont__native_notool__r20",
            adapter_format="peft_adapter",
            phase_one_priority=1,
        ),
        ModelSpec(
            key="qwen_util_tool",
            architecture_family="qwen35_9b",
            objective="utilitarian",
            inference_condition="native_tool",
            repo_id="agentic-moral-alignment/qwen35-9b__ipd_str_tft__util__native_tool__r1",
            adapter_format="full_or_trainer_export",
            phase_one_priority=1,
        ),
        ModelSpec(
            key="qwen_game_tool",
            architecture_family="qwen35_9b",
            objective="game_theoretic",
            inference_condition="native_tool",
            repo_id="agentic-moral-alignment/qwen35-9b__ipd_str_tft__game__native_tool__r1",
            adapter_format="full_or_trainer_export",
            phase_one_priority=1,
        ),
    ]


def getCheckpointPolicy() -> CheckpointPolicy:
    return CheckpointPolicy(
        name="behavior_anchored_sparse_ladder",
        checkpoint_count=5,
        target_events=(
            "before_behavior_shift",
            "first_clear_behavior_shift",
            "mid_training",
            "late_training",
            "final_checkpoint",
        ),
        selection_rule=(
            "Use the checkpoints or runs dataset to choose five snapshots per model family. "
            "Prefer behavior-anchored checkpoints over evenly spaced steps when eval logs exist; "
            "fall back to approximately early, early-mid, mid, late, and final if only step numbers are available."
        ),
    )
