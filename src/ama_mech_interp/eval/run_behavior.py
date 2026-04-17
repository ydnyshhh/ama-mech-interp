from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..run_spec import RunSpec, buildRunKey, getOutputDirectory
from ..storage import readJsonlFile, writeJsonFile, writeJsonlFile


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
                status="observed" if observed_row else "planned",
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
