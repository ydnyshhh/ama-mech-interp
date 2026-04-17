from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..storage import writeJsonFile


@dataclass(frozen=True)
class ProbeJobSpec:
    job_name: str
    probe_targets: tuple[str, ...]
    prompt_suite_path: str
    activation_plan_paths: tuple[str, ...]
    output_directory: str


def writeProbeJobSpec(
    output_path: Path,
    prompt_suite_path: Path,
    activation_plan_paths: list[Path],
) -> Path:
    payload = ProbeJobSpec(
        job_name="linear_probe_sweep",
        probe_targets=("chosen_action", "game_family", "history_type"),
        prompt_suite_path=str(prompt_suite_path),
        activation_plan_paths=tuple(str(path) for path in activation_plan_paths),
        output_directory=str(output_path.parent),
    )
    return writeJsonFile(output_path, payload)
