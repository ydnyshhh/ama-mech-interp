from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..storage import writeJsonFile


@dataclass(frozen=True)
class DriftJobSpec:
    job_name: str
    metric_names: tuple[str, ...]
    prompt_suite_path: str
    activation_plan_paths: tuple[str, ...]
    output_directory: str


def writeDriftJobSpec(
    output_path: Path,
    prompt_suite_path: Path,
    activation_plan_paths: list[Path],
) -> Path:
    payload = DriftJobSpec(
        job_name="layerwise_representation_drift",
        metric_names=("cka", "cosine_drift", "procrustes_distance"),
        prompt_suite_path=str(prompt_suite_path),
        activation_plan_paths=tuple(str(path) for path in activation_plan_paths),
        output_directory=str(output_path.parent),
    )
    return writeJsonFile(output_path, payload)
