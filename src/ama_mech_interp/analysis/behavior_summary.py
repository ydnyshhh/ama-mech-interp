from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..storage import readJsonlFile, writeJsonFile


@dataclass(frozen=True)
class BehaviorSummaryJobSpec:
    job_name: str
    prompt_suite_path: str
    behavior_rows_paths: tuple[str, ...]
    output_directory: str


def buildBehaviorSummary(
    prompt_suite_path: Path,
    behavior_rows_path: Path,
) -> dict[str, object]:
    prompt_suite_rows = {str(row["prompt_id"]): row for row in readJsonlFile(prompt_suite_path)}
    behavior_rows = readJsonlFile(behavior_rows_path)
    action_counts: dict[str, Counter[str]] = defaultdict(Counter)
    game_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for behavior_row in behavior_rows:
        if behavior_row.get("chosen_action") is None:
            continue
        model_key = str(behavior_row["model_key"])
        prompt_row = prompt_suite_rows[str(behavior_row["prompt_id"])]
        subset_key = f"{model_key}::{prompt_row['subset']}"
        game_key = f"{model_key}::{prompt_row['formal_game']}"
        action_counts[subset_key][str(behavior_row["chosen_action"])] += 1
        game_counts[game_key][str(behavior_row["chosen_action"])] += 1
    return {
        "action_distribution_by_subset": {key: dict(counter) for key, counter in action_counts.items()},
        "action_distribution_by_game": {key: dict(counter) for key, counter in game_counts.items()},
    }


def writeBehaviorSummary(output_path: Path, prompt_suite_path: Path, behavior_rows_path: Path) -> Path:
    summary = buildBehaviorSummary(prompt_suite_path, behavior_rows_path)
    return writeJsonFile(output_path, summary)


def writeBehaviorSummaryJobSpec(output_path: Path, prompt_suite_path: Path, behavior_rows_paths: list[Path]) -> Path:
    payload = BehaviorSummaryJobSpec(
        job_name="behavioral_summary",
        prompt_suite_path=str(prompt_suite_path),
        behavior_rows_paths=tuple(str(path) for path in behavior_rows_paths),
        output_directory=str(output_path.parent),
    )
    return writeJsonFile(output_path, payload)
