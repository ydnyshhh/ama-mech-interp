from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..storage import writeJsonFile


@dataclass(frozen=True)
class LogitDepthJobSpec:
    job_name: str
    prompt_suite_path: str
    activation_plan_paths: tuple[str, ...]
    target_action_field: str
    output_directory: str


def writeLogitDepthJobSpec(
    output_path: Path,
    prompt_suite_path: Path,
    activation_plan_paths: list[Path],
) -> Path:
    payload = LogitDepthJobSpec(
        job_name="logit_through_depth",
        prompt_suite_path=str(prompt_suite_path),
        activation_plan_paths=tuple(str(path) for path in activation_plan_paths),
        target_action_field="chosen_action",
        output_directory=str(output_path.parent),
    )
    return writeJsonFile(output_path, payload)
