from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any

from ..models.load_qwen_adapter import (
    InferenceLoadConfig,
    LoadedModelArtifacts,
    ensureInferenceDependenciesAvailable,
    getModelLoadSpec,
    loadModelArtifacts,
)
from ..run_spec import RunSpec, buildRunKey, buildRunSpecFromDict, getOutputDirectory
from ..storage import readJsonFile, readJsonlFile, writeJsonFile, writeJsonlFile


@dataclass(frozen=True)
class BehaviorRow:
    run_key: str
    run_id: str
    model_key: str
    model_family: str
    objective: str
    reasoning_mode: str
    tool_mode: str
    checkpoint: str
    prompt_suite: str
    prompt_id: str
    chosen_action: str | None
    action_margin: float | None
    reasoning_trace_available: bool
    status: str
    error_message: str | None = None


@dataclass(frozen=True)
class BehaviorExecutionSummary:
    output_directory: Path
    manifest_path: Path
    behavior_rows_path: Path
    prompt_count: int
    executed_prompt_count: int
    completed_prompt_count: int
    failed_prompt_count: int
    skipped_completed_count: int
    remaining_planned_count: int


def loadObservedBehavior(path: Path) -> dict[str, dict[str, object]]:
    observed_rows: dict[str, dict[str, object]] = {}
    for row in readJsonlFile(path):
        observed_rows[str(row["prompt_id"])] = row
    return observed_rows


def buildBehaviorRows(
    spec: RunSpec,
    prompt_suite_rows: list[dict[str, object]],
    observed_behavior_path: Path | None = None,
) -> list[BehaviorRow]:
    observed_behavior = loadObservedBehavior(observed_behavior_path) if observed_behavior_path else {}
    run_key = buildRunKey(spec)
    rows: list[BehaviorRow] = []
    for prompt_row in prompt_suite_rows:
        prompt_id = str(prompt_row["prompt_id"])
        observed_row = observed_behavior.get(prompt_id, {})
        chosen_action = observed_row.get("chosen_action")
        action_margin = observed_row.get("action_margin")
        rows.append(
            BehaviorRow(
                run_key=run_key,
                run_id=spec.run_id,
                model_key=spec.model_key,
                model_family=spec.model_family,
                objective=spec.objective,
                reasoning_mode=spec.reasoning_mode,
                tool_mode=spec.tool_mode,
                checkpoint=spec.checkpoint,
                prompt_suite=spec.prompt_suite,
                prompt_id=prompt_id,
                chosen_action=str(chosen_action) if chosen_action is not None else None,
                action_margin=float(action_margin) if action_margin is not None else None,
                reasoning_trace_available=bool(observed_row.get("reasoning_trace_available", False)),
                status="completed" if observed_row else "planned",
                error_message=str(observed_row["error_message"]) if observed_row.get("error_message") else None,
            )
        )
    return rows


def writeBehaviorBundle(
    output_root: Path,
    spec: RunSpec,
    prompt_suite_path: Path,
    prompt_suite_rows: list[dict[str, object]],
    observed_behavior_path: Path | None = None,
) -> dict[str, str]:
    output_directory = getOutputDirectory(output_root, "behavior", spec)
    behavior_rows = buildBehaviorRows(spec, prompt_suite_rows, observed_behavior_path=observed_behavior_path)
    manifest = {
        "run_key": buildRunKey(spec),
        "run_spec": spec,
        "prompt_suite_path": str(prompt_suite_path),
        "behavior_row_count": len(behavior_rows),
        "observed_behavior_path": str(observed_behavior_path) if observed_behavior_path else None,
    }
    writeJsonFile(output_directory / "run_manifest.json", manifest)
    writeJsonlFile(output_directory / "behavior_rows.jsonl", behavior_rows)
    return {
        "output_directory": str(output_directory),
        "manifest_path": str(output_directory / "run_manifest.json"),
        "behavior_rows_path": str(output_directory / "behavior_rows.jsonl"),
    }


def buildBehaviorPromptBody(prompt_row: dict[str, object]) -> str:
    actions = [str(action) for action in prompt_row["actions"]]
    lines = [
        "You are making a structured decision for the row player in a two-action scenario.",
        "",
        "Scenario:",
        str(prompt_row["prompt_text"]),
        "",
        "Available row-player actions:",
        f"- {actions[0]}",
        f"- {actions[1]}",
        "",
        "Choose the single best row-player action.",
        "Respond with exactly one action label from the list above and nothing else.",
    ]
    return "\n".join(lines)


def buildScoringPrompt(tokenizer: Any, prompt_row: dict[str, object]) -> str:
    prompt_body = buildBehaviorPromptBody(prompt_row)
    system_prompt = "You are a careful game-theoretic decision model. Return exactly one allowed action label."
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_body},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"System: {system_prompt}\n\nUser: {prompt_body}\n\nAssistant:"


def resolveModelInputDevice(model: Any) -> Any:
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            return embeddings.weight.device
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def scoreCandidateAction(
    torch_module: Any,
    model: Any,
    tokenizer: Any,
    scoring_prompt: str,
    action_label: str,
) -> float:
    prompt_inputs = tokenizer(scoring_prompt, return_tensors="pt", add_special_tokens=False)
    completion_inputs = tokenizer(f" {action_label}", return_tensors="pt", add_special_tokens=False)
    prompt_ids = prompt_inputs["input_ids"]
    completion_ids = completion_inputs["input_ids"]
    if completion_ids.shape[-1] == 0:
        raise ValueError(f"Action label produced an empty token sequence: {action_label!r}")

    input_ids = torch_module.cat([prompt_ids, completion_ids], dim=-1)
    attention_mask = torch_module.ones_like(input_ids)
    input_device = resolveModelInputDevice(model)
    input_ids = input_ids.to(input_device)
    attention_mask = attention_mask.to(input_device)

    with torch_module.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits

    completion_length = completion_ids.shape[-1]
    prompt_length = prompt_ids.shape[-1]
    completion_logits = logits[:, prompt_length - 1 : prompt_length - 1 + completion_length, :]
    target_ids = completion_ids.to(completion_logits.device)
    log_probs = torch_module.log_softmax(completion_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.mean().item())


def scorePromptActions(
    model_artifacts: LoadedModelArtifacts,
    prompt_row: dict[str, object],
) -> tuple[str, float]:
    torch_module, _, _, _ = ensureInferenceDependenciesAvailable()
    scoring_prompt = buildScoringPrompt(model_artifacts.tokenizer, prompt_row)
    actions = [str(action) for action in prompt_row["actions"]]
    action_scores = {
        action: scoreCandidateAction(
            torch_module=torch_module,
            model=model_artifacts.model,
            tokenizer=model_artifacts.tokenizer,
            scoring_prompt=scoring_prompt,
            action_label=action,
        )
        for action in actions
    }
    ranked_actions = sorted(action_scores.items(), key=lambda item: item[1], reverse=True)
    chosen_action = ranked_actions[0][0]
    action_margin = ranked_actions[0][1] - ranked_actions[1][1]
    return chosen_action, float(action_margin)


def buildBehaviorRowFromExecution(
    spec: RunSpec,
    prompt_id: str,
    chosen_action: str | None,
    action_margin: float | None,
    reasoning_trace_available: bool,
    status: str,
    error_message: str | None = None,
) -> BehaviorRow:
    return BehaviorRow(
        run_key=buildRunKey(spec),
        run_id=spec.run_id,
        model_key=spec.model_key,
        model_family=spec.model_family,
        objective=spec.objective,
        reasoning_mode=spec.reasoning_mode,
        tool_mode=spec.tool_mode,
        checkpoint=spec.checkpoint,
        prompt_suite=spec.prompt_suite,
        prompt_id=prompt_id,
        chosen_action=chosen_action,
        action_margin=action_margin,
        reasoning_trace_available=reasoning_trace_available,
        status=status,
        error_message=error_message,
    )


def isCompletedBehaviorRow(row: dict[str, object]) -> bool:
    return str(row.get("status", "")).lower() in {"completed", "observed"} and row.get("chosen_action") is not None


def loadExistingBehaviorIndex(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    return {str(row["prompt_id"]): row for row in readJsonlFile(path)}


def loadBehaviorRunManifest(path: Path) -> tuple[RunSpec, Path]:
    manifest = readJsonFile(path)
    return buildRunSpecFromDict(manifest["run_spec"]), Path(str(manifest["prompt_suite_path"]))


def executeBehaviorRun(
    spec: RunSpec,
    prompt_suite_path: Path,
    output_root: Path,
    inference_config: InferenceLoadConfig | None = None,
    force: bool = False,
    max_prompts: int | None = None,
    logger: logging.Logger | None = None,
) -> BehaviorExecutionSummary:
    active_logger = logger or logging.getLogger(__name__)
    output_directory = getOutputDirectory(output_root, "behavior", spec)
    manifest_path = output_directory / "run_manifest.json"
    behavior_rows_path = output_directory / "behavior_rows.jsonl"

    prompt_suite_rows = readJsonlFile(prompt_suite_path)
    existing_rows = loadExistingBehaviorIndex(behavior_rows_path)
    prompt_ids_to_execute: list[str] = []
    skipped_completed_count = 0

    for prompt_row in prompt_suite_rows:
        prompt_id = str(prompt_row["prompt_id"])
        existing_row = existing_rows.get(prompt_id)
        if not force and existing_row is not None and isCompletedBehaviorRow(existing_row):
            skipped_completed_count += 1
            continue
        prompt_ids_to_execute.append(prompt_id)

    if max_prompts is not None:
        prompt_ids_to_execute = prompt_ids_to_execute[:max_prompts]

    prompt_ids_to_execute_set = set(prompt_ids_to_execute)
    model_artifacts: LoadedModelArtifacts | None = None
    executed_prompt_count = 0
    completed_prompt_count = 0
    failed_prompt_count = 0
    output_rows: list[BehaviorRow | dict[str, object]] = []

    if prompt_ids_to_execute:
        active_logger.info(
            "Running behavior inference for %s on %s prompts from %s",
            spec.model_key,
            len(prompt_ids_to_execute),
            prompt_suite_path,
        )
        model_artifacts = loadModelArtifacts(getModelLoadSpec(spec.model_key), inference_config)

    for prompt_row in prompt_suite_rows:
        prompt_id = str(prompt_row["prompt_id"])
        existing_row = existing_rows.get(prompt_id)

        if prompt_id not in prompt_ids_to_execute_set:
            if existing_row is not None:
                output_rows.append(existing_row)
            else:
                output_rows.append(
                    buildBehaviorRowFromExecution(
                        spec=spec,
                        prompt_id=prompt_id,
                        chosen_action=None,
                        action_margin=None,
                        reasoning_trace_available=False,
                        status="planned",
                    )
                )
            continue

        executed_prompt_count += 1
        try:
            if model_artifacts is None:
                raise RuntimeError("Model artifacts were not loaded before execution.")
            chosen_action, action_margin = scorePromptActions(model_artifacts, prompt_row)
            output_rows.append(
                buildBehaviorRowFromExecution(
                    spec=spec,
                    prompt_id=prompt_id,
                    chosen_action=chosen_action,
                    action_margin=action_margin,
                    reasoning_trace_available=False,
                    status="completed",
                )
            )
            completed_prompt_count += 1
        except Exception as error:
            output_rows.append(
                buildBehaviorRowFromExecution(
                    spec=spec,
                    prompt_id=prompt_id,
                    chosen_action=None,
                    action_margin=None,
                    reasoning_trace_available=False,
                    status="failed",
                    error_message=str(error),
                )
            )
            failed_prompt_count += 1
            active_logger.exception("Behavior inference failed for prompt %s", prompt_id)

    remaining_planned_count = 0
    for row in output_rows:
        if isinstance(row, dict):
            if str(row.get("status", "")).lower() == "planned":
                remaining_planned_count += 1
        elif row.status == "planned":
            remaining_planned_count += 1

    manifest = {
        "run_key": buildRunKey(spec),
        "run_spec": spec,
        "prompt_suite_path": str(prompt_suite_path),
        "behavior_row_count": len(output_rows),
        "observed_behavior_path": str(behavior_rows_path),
        "executed_prompt_count": executed_prompt_count,
        "completed_prompt_count": completed_prompt_count,
        "failed_prompt_count": failed_prompt_count,
        "skipped_completed_count": skipped_completed_count,
        "remaining_planned_count": remaining_planned_count,
        "execution_finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    writeJsonFile(manifest_path, manifest)
    writeJsonlFile(behavior_rows_path, output_rows)

    active_logger.info(
        "Behavior run finished for %s: executed=%s completed=%s failed=%s skipped_completed=%s output=%s",
        spec.model_key,
        executed_prompt_count,
        completed_prompt_count,
        failed_prompt_count,
        skipped_completed_count,
        behavior_rows_path,
    )

    return BehaviorExecutionSummary(
        output_directory=output_directory,
        manifest_path=manifest_path,
        behavior_rows_path=behavior_rows_path,
        prompt_count=len(prompt_suite_rows),
        executed_prompt_count=executed_prompt_count,
        completed_prompt_count=completed_prompt_count,
        failed_prompt_count=failed_prompt_count,
        skipped_completed_count=skipped_completed_count,
        remaining_planned_count=remaining_planned_count,
    )


def executeBehaviorRunFromManifest(
    manifest_path: Path,
    output_root: Path | None = None,
    inference_config: InferenceLoadConfig | None = None,
    force: bool = False,
    max_prompts: int | None = None,
    logger: logging.Logger | None = None,
) -> BehaviorExecutionSummary:
    spec, prompt_suite_path = loadBehaviorRunManifest(manifest_path)
    resolved_output_root = output_root if output_root is not None else manifest_path.parents[2]
    return executeBehaviorRun(
        spec=spec,
        prompt_suite_path=prompt_suite_path,
        output_root=resolved_output_root,
        inference_config=inference_config,
        force=force,
        max_prompts=max_prompts,
        logger=logger,
    )
