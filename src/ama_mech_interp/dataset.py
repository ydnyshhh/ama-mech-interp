from __future__ import annotations

import ast
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


GAME_NORMALIZATION = {
    "prisoner's dilemma": "prisoners_dilemma",
    "stag hunt": "stag_hunt",
    "chicken": "chicken",
    "bach or stravinski": "bach_or_stravinski",
    "coordination": "coordination",
    "matching pennies": "matching_pennies",
    "no conflict": "no_conflict",
}


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: int
    original_game_family: str
    game_family: str
    prompt_text: str
    row_actions: tuple[str, ...]
    column_actions: tuple[str, ...]
    payoff_matrix: tuple[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[tuple[int, int], tuple[int, int]],
    ]
    risk_level: int | None
    quality_score: float | None
    equilibria_score: float | None
    target_nash_equilibria: str
    target_utility_maximizing: str
    target_rawlsian: str
    target_nash_social_welfare: str


@dataclass(frozen=True)
class DatasetSummary:
    row_count: int
    missing_story_count: int
    game_counts: dict[str, int]
    risk_counts: dict[str, int]
    disagreement_counts: dict[str, int]


def normalizeGameFamily(value: str) -> str:
    cleaned_value = value.strip().lower()
    return GAME_NORMALIZATION.get(cleaned_value, cleaned_value.replace(" ", "_"))


def parseActionList(value: str) -> tuple[str, ...]:
    parsed_value = ast.literal_eval(value)
    return tuple(str(item).strip() for item in parsed_value)


def parsePayoffCell(value: str) -> tuple[int, int]:
    parsed_value = ast.literal_eval(value)
    return int(parsed_value[0]), int(parsed_value[1])


def parseOptionalFloat(value: str) -> float | None:
    cleaned_value = value.strip()
    if not cleaned_value:
        return None
    return float(cleaned_value)


def parseOptionalInt(value: str) -> int | None:
    cleaned_value = value.strip()
    if not cleaned_value:
        return None
    return int(cleaned_value)


def classifyDisagreement(record: PromptRecord) -> str:
    target_map = {
        "nash": record.target_nash_equilibria,
        "util": record.target_utility_maximizing,
        "rawls": record.target_rawlsian,
        "nsw": record.target_nash_social_welfare,
    }
    unique_target_count = len(set(target_map.values()))
    normative_target_count = len(
        {
            record.target_utility_maximizing,
            record.target_rawlsian,
            record.target_nash_social_welfare,
        }
    )
    if unique_target_count == 1:
        return "all_agree"
    if normative_target_count == 1 and record.target_nash_equilibria != record.target_utility_maximizing:
        return "nash_vs_normative"
    if normative_target_count > 1:
        return "normative_split"
    return "mixed"


def loadPromptRecords(csv_path: Path) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt_text = row["story_row"].strip()
            if not prompt_text:
                continue
            payoff_matrix = (
                (
                    parsePayoffCell(row["1_1_payoff"]),
                    parsePayoffCell(row["1_2_payoff"]),
                ),
                (
                    parsePayoffCell(row["2_1_payoff"]),
                    parsePayoffCell(row["2_2_payoff"]),
                ),
            )
            records.append(
                PromptRecord(
                    prompt_id=int(row["id"]),
                    original_game_family=row["formal_game"].strip(),
                    game_family=normalizeGameFamily(row["formal_game"]),
                    prompt_text=prompt_text,
                    row_actions=parseActionList(row["actions_row"]),
                    column_actions=parseActionList(row["actions_column"]),
                    payoff_matrix=payoff_matrix,
                    risk_level=parseOptionalInt(row["risk_level"]),
                    quality_score=parseOptionalFloat(row["quality_score"]),
                    equilibria_score=parseOptionalFloat(row["equilibria_score"]),
                    target_nash_equilibria=row["target_nash_equilibria"].strip(),
                    target_utility_maximizing=row["target_utility_maximizing"].strip(),
                    target_rawlsian=row["target_rawlsian"].strip(),
                    target_nash_social_welfare=row["target_nash_social_welfare"].strip(),
                )
            )
    return records


def summarizeDataset(records: list[PromptRecord]) -> DatasetSummary:
    game_counter: Counter[str] = Counter()
    risk_counter: Counter[str] = Counter()
    disagreement_counter: Counter[str] = Counter()

    for record in records:
        game_counter[record.original_game_family] += 1
        risk_key = str(record.risk_level) if record.risk_level is not None else "missing"
        risk_counter[risk_key] += 1
        disagreement_counter[classifyDisagreement(record)] += 1

    return DatasetSummary(
        row_count=len(records),
        missing_story_count=0,
        game_counts=dict(game_counter.most_common()),
        risk_counts=dict(risk_counter.most_common()),
        disagreement_counts=dict(disagreement_counter.most_common()),
    )
