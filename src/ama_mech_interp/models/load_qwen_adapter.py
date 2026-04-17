from __future__ import annotations

from dataclasses import dataclass

from ..run_spec import RunSpec


BASE_MODEL_REPO = "unsloth/Qwen3.5-9B"


@dataclass(frozen=True)
class ModelLoadSpec:
    model_key: str
    model_family: str
    objective: str
    reasoning_mode: str
    tool_mode: str
    checkpoint: str
    repo_id: str
    adapter_format: str
    base_model_repo: str


def getPhaseOneModelLoadSpecs() -> list[ModelLoadSpec]:
    return [
        ModelLoadSpec(
            model_key="qwen_base",
            model_family="qwen35_9b",
            objective="base",
            reasoning_mode="native",
            tool_mode="none",
            checkpoint="base",
            repo_id=BASE_MODEL_REPO,
            adapter_format="base_model",
            base_model_repo=BASE_MODEL_REPO,
        ),
        ModelLoadSpec(
            model_key="qwen_deont_tool",
            model_family="qwen35_9b",
            objective="deontological",
            reasoning_mode="native",
            tool_mode="tool",
            checkpoint="final",
            repo_id="agentic-moral-alignment/qwen35-9b__ipd_str_tft__deont__native_tool__r1",
            adapter_format="peft_adapter",
            base_model_repo=BASE_MODEL_REPO,
        ),
        ModelLoadSpec(
            model_key="qwen_deont_notool",
            model_family="qwen35_9b",
            objective="deontological",
            reasoning_mode="native",
            tool_mode="notool",
            checkpoint="final",
            repo_id="agentic-moral-alignment/qwen35-9b__ipd_str_tft__deont__native_notool__r20",
            adapter_format="peft_adapter",
            base_model_repo=BASE_MODEL_REPO,
        ),
        ModelLoadSpec(
            model_key="qwen_util_tool",
            model_family="qwen35_9b",
            objective="utilitarian",
            reasoning_mode="native",
            tool_mode="tool",
            checkpoint="final",
            repo_id="agentic-moral-alignment/qwen35-9b__ipd_str_tft__util__native_tool__r1",
            adapter_format="full_or_trainer_export",
            base_model_repo=BASE_MODEL_REPO,
        ),
        ModelLoadSpec(
            model_key="qwen_game_tool",
            model_family="qwen35_9b",
            objective="game_theoretic",
            reasoning_mode="native",
            tool_mode="tool",
            checkpoint="final",
            repo_id="agentic-moral-alignment/qwen35-9b__ipd_str_tft__game__native_tool__r1",
            adapter_format="full_or_trainer_export",
            base_model_repo=BASE_MODEL_REPO,
        ),
    ]


def getMinimumViableModelLoadSpecs() -> list[ModelLoadSpec]:
    preferred_keys = {"qwen_base", "qwen_deont_tool", "qwen_util_tool", "qwen_game_tool"}
    return [spec for spec in getPhaseOneModelLoadSpecs() if spec.model_key in preferred_keys]


def getModelLoadSpec(model_key: str) -> ModelLoadSpec:
    for spec in getPhaseOneModelLoadSpecs():
        if spec.model_key == model_key:
            return spec
    raise KeyError(f"Unknown model key: {model_key}")


def buildRunSpec(model_spec: ModelLoadSpec, prompt_suite: str, run_id: str) -> RunSpec:
    return RunSpec(
        run_id=run_id,
        model_key=model_spec.model_key,
        model_family=model_spec.model_family,
        objective=model_spec.objective,
        reasoning_mode=model_spec.reasoning_mode,
        tool_mode=model_spec.tool_mode,
        checkpoint=model_spec.checkpoint,
        prompt_suite=prompt_suite,
        repo_id=model_spec.repo_id,
    )
