from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from ..storage import readJsonlFile, writeJsonFile, writeJsonlFile


@dataclass(frozen=True)
class DisagreementRow:
    prompt_id: str
    formal_game: str
    subset: str
    distinct_action_count: int
    model_count: int
    target_disagreement_count: int
    action_margin_span: float
    disagreement_score: float


@dataclass(frozen=True)
class DisagreementJobSpec:
    job_name: str
    prompt_suite_path: str
    behavior_rows_paths: tuple[str, ...]
    output_directory: str


def countTargetDisagreement(prompt_row: dict[str, object]) -> int:
    values = {
        str(prompt_row["target_nash"]),
        str(prompt_row["target_util"]),
        str(prompt_row["target_rawls"]),
        str(prompt_row["target_nsw"]),
    }
    return max(len(values) - 1, 0)


def buildDisagreementRows(
    prompt_suite_path: Path,
    behavior_rows_paths: list[Path],
) -> list[DisagreementRow]:
    prompt_suite_rows = {str(row["prompt_id"]): row for row in readJsonlFile(prompt_suite_path)}
    grouped_rows: dict[str, list[dict[str, object]]] = {}
    for behavior_rows_path in behavior_rows_paths:
        for behavior_row in readJsonlFile(behavior_rows_path):
            if behavior_row.get("chosen_action") is None:
                continue
            prompt_id = str(behavior_row["prompt_id"])
            grouped_rows.setdefault(prompt_id, []).append(behavior_row)

    disagreement_rows: list[DisagreementRow] = []
    for prompt_id, rows in grouped_rows.items():
        prompt_row = prompt_suite_rows[prompt_id]
        chosen_actions = {str(row["chosen_action"]) for row in rows if row.get("chosen_action") is not None}
        action_margins = [float(row["action_margin"]) for row in rows if row.get("action_margin") is not None]
        action_margin_span = max(action_margins) - min(action_margins) if len(action_margins) >= 2 else 0.0
        target_disagreement_count = countTargetDisagreement(prompt_row)
        disagreement_score = (
            float(len(chosen_actions) - 1) * 2.0
            + float(target_disagreement_count) * 1.5
            + action_margin_span
        )
        disagreement_rows.append(
            DisagreementRow(
                prompt_id=prompt_id,
                formal_game=str(prompt_row["formal_game"]),
                subset=str(prompt_row["subset"]),
                distinct_action_count=len(chosen_actions),
                model_count=len(rows),
                target_disagreement_count=target_disagreement_count,
                action_margin_span=action_margin_span,
                disagreement_score=disagreement_score,
            )
        )
    disagreement_rows.sort(key=lambda row: (row.disagreement_score, row.prompt_id), reverse=True)
    return disagreement_rows


def buildPairwiseDisagreementMatrix(behavior_rows_paths: list[Path]) -> dict[str, int]:
    behavior_by_model: dict[str, dict[str, str]] = {}
    for behavior_rows_path in behavior_rows_paths:
        for behavior_row in readJsonlFile(behavior_rows_path):
            if behavior_row.get("chosen_action") is None:
                continue
            model_key = str(behavior_row["model_key"])
            prompt_id = str(behavior_row["prompt_id"])
            behavior_by_model.setdefault(model_key, {})[prompt_id] = str(behavior_row["chosen_action"])

    matrix: dict[str, int] = {}
    for left_key, right_key in combinations(sorted(behavior_by_model), 2):
        shared_prompt_ids = set(behavior_by_model[left_key]).intersection(behavior_by_model[right_key])
        disagreement_count = 0
        for prompt_id in shared_prompt_ids:
            if behavior_by_model[left_key][prompt_id] != behavior_by_model[right_key][prompt_id]:
                disagreement_count += 1
        matrix[f"{left_key}__vs__{right_key}"] = disagreement_count
    return matrix


def writeDisagreementArtifacts(
    output_directory: Path,
    prompt_suite_path: Path,
    behavior_rows_paths: list[Path],
) -> dict[str, str]:
    disagreement_rows = buildDisagreementRows(prompt_suite_path, behavior_rows_paths)
    pairwise_matrix = buildPairwiseDisagreementMatrix(behavior_rows_paths)
    writeJsonlFile(output_directory / "ranked_prompts.jsonl", disagreement_rows)
    writeJsonFile(output_directory / "pairwise_matrix.json", pairwise_matrix)
    return {
        "ranked_prompts_path": str(output_directory / "ranked_prompts.jsonl"),
        "pairwise_matrix_path": str(output_directory / "pairwise_matrix.json"),
    }


def writeDisagreementJobSpec(output_path: Path, prompt_suite_path: Path, behavior_rows_paths: list[Path]) -> Path:
    payload = DisagreementJobSpec(
        job_name="behavioral_disagreement_mining",
        prompt_suite_path=str(prompt_suite_path),
        behavior_rows_paths=tuple(str(path) for path in behavior_rows_paths),
        output_directory=str(output_path.parent),
    )
    return writeJsonFile(output_path, payload)
