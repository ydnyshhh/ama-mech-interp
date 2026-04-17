from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    model_key: str
    model_family: str
    objective: str
    reasoning_mode: str
    tool_mode: str
    checkpoint: str
    prompt_suite: str
    repo_id: str


def buildRunKey(spec: RunSpec) -> str:
    return "__".join(
        [
            spec.model_family,
            spec.objective,
            spec.reasoning_mode,
            spec.tool_mode,
            spec.checkpoint,
            spec.prompt_suite,
            spec.run_id,
        ]
    )


def getOutputDirectory(output_root: Path, stage_name: str, spec: RunSpec) -> Path:
    return output_root / stage_name / buildRunKey(spec)


def getStageFilePath(output_root: Path, stage_name: str, spec: RunSpec, file_name: str) -> Path:
    return getOutputDirectory(output_root, stage_name, spec) / file_name
