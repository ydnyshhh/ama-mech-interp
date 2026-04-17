from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class InferenceLoadConfig:
    device_map: str | None = "auto"
    torch_dtype: str = "auto"
    local_files_only: bool = False
    trust_remote_code: bool = True
    load_format_fallback: bool = True


@dataclass(frozen=True)
class LoadedModelArtifacts:
    model: Any
    tokenizer: Any
    model_spec: ModelLoadSpec


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


def ensureInferenceDependenciesAvailable() -> tuple[Any, Any, Any, Any | None]:
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "Inference dependencies are not installed. Run `uv sync --extra inference` before `run-behavior`."
        ) from error

    try:
        from peft import PeftModel  # type: ignore
    except ImportError:
        PeftModel = None

    return torch, AutoModelForCausalLM, AutoTokenizer, PeftModel


def resolveTorchDtype(torch_module: Any, torch_dtype: str) -> Any:
    dtype_name = torch_dtype.lower()
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "float32":
        return torch_module.float32
    raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def buildTokenizerLoadKwargs(config: InferenceLoadConfig) -> dict[str, Any]:
    return {
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
    }


def buildModelLoadKwargs(torch_module: Any, config: InferenceLoadConfig) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
        "low_cpu_mem_usage": True,
    }
    resolved_dtype = resolveTorchDtype(torch_module, config.torch_dtype)
    model_kwargs["torch_dtype"] = resolved_dtype
    if config.device_map is not None:
        model_kwargs["device_map"] = config.device_map
    return model_kwargs


def prepareTokenizerForInference(tokenizer: Any) -> Any:
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def loadBaseModel(model_spec: ModelLoadSpec, config: InferenceLoadConfig) -> LoadedModelArtifacts:
    torch_module, AutoModelForCausalLM, AutoTokenizer, _ = ensureInferenceDependenciesAvailable()
    tokenizer = prepareTokenizerForInference(
        AutoTokenizer.from_pretrained(model_spec.base_model_repo, **buildTokenizerLoadKwargs(config))
    )
    model = AutoModelForCausalLM.from_pretrained(model_spec.repo_id, **buildModelLoadKwargs(torch_module, config))
    model.eval()
    return LoadedModelArtifacts(model=model, tokenizer=tokenizer, model_spec=model_spec)


def loadPeftAdapterModel(model_spec: ModelLoadSpec, config: InferenceLoadConfig) -> LoadedModelArtifacts:
    torch_module, AutoModelForCausalLM, AutoTokenizer, PeftModel = ensureInferenceDependenciesAvailable()
    if PeftModel is None:
        raise RuntimeError(
            "PEFT is required for adapter-backed models. Run `uv sync --extra inference` before `run-behavior`."
        )

    tokenizer = prepareTokenizerForInference(
        AutoTokenizer.from_pretrained(model_spec.base_model_repo, **buildTokenizerLoadKwargs(config))
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_spec.base_model_repo,
        **buildModelLoadKwargs(torch_module, config),
    )
    model = PeftModel.from_pretrained(
        base_model,
        model_spec.repo_id,
        local_files_only=config.local_files_only,
    )
    model.eval()
    return LoadedModelArtifacts(model=model, tokenizer=tokenizer, model_spec=model_spec)


def loadFullOrTrainerExportModel(model_spec: ModelLoadSpec, config: InferenceLoadConfig) -> LoadedModelArtifacts:
    torch_module, AutoModelForCausalLM, AutoTokenizer, PeftModel = ensureInferenceDependenciesAvailable()
    tokenizer = prepareTokenizerForInference(
        AutoTokenizer.from_pretrained(model_spec.base_model_repo, **buildTokenizerLoadKwargs(config))
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_spec.repo_id,
            **buildModelLoadKwargs(torch_module, config),
        )
        model.eval()
        return LoadedModelArtifacts(model=model, tokenizer=tokenizer, model_spec=model_spec)
    except Exception:
        if not config.load_format_fallback or PeftModel is None:
            raise

    base_model = AutoModelForCausalLM.from_pretrained(
        model_spec.base_model_repo,
        **buildModelLoadKwargs(torch_module, config),
    )
    model = PeftModel.from_pretrained(
        base_model,
        model_spec.repo_id,
        local_files_only=config.local_files_only,
    )
    model.eval()
    return LoadedModelArtifacts(model=model, tokenizer=tokenizer, model_spec=model_spec)


def loadModelArtifacts(model_spec: ModelLoadSpec, config: InferenceLoadConfig | None = None) -> LoadedModelArtifacts:
    load_config = config or InferenceLoadConfig()
    if model_spec.adapter_format == "base_model":
        return loadBaseModel(model_spec, load_config)
    if model_spec.adapter_format == "peft_adapter":
        return loadPeftAdapterModel(model_spec, load_config)
    if model_spec.adapter_format == "full_or_trainer_export":
        return loadFullOrTrainerExportModel(model_spec, load_config)
    raise ValueError(f"Unsupported adapter format: {model_spec.adapter_format}")
