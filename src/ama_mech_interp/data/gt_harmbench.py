from __future__ import annotations

import ast
import csv
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


DISAGREEMENT_PRIORITY = {
    "normative_split": 3,
    "nash_vs_normative": 2,
    "mixed": 1,
    "all_agree": 0,
}


@dataclass(frozen=True)
class GtHarmBenchRow:
    prompt_id: str
    source: str
    formal_game: str
    normalized_game: str
    prompt_text: str
    story_column_text: str
    actions_row: tuple[str, ...]
    actions_column: tuple[str, ...]
    payoff_matrix: tuple[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[tuple[int, int], tuple[int, int]],
    ]
    risk_level: int | None
    quality_score: float | None
    equilibria_score: float | None
    target_nash: str
    target_util: str
    target_rawls: str
    target_nsw: str
    disagreement_bucket: str
    disagreement_priority: int


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


def classifyDisagreement(target_nash: str, target_util: str, target_rawls: str, target_nsw: str) -> str:
    target_values = {target_nash, target_util, target_rawls, target_nsw}
    normative_values = {target_util, target_rawls, target_nsw}
    if len(target_values) == 1:
        return "all_agree"
    if len(normative_values) == 1 and target_nash != target_util:
        return "nash_vs_normative"
    if len(normative_values) > 1:
        return "normative_split"
    return "mixed"


def loadGtHarmBenchRows(csv_path: Path) -> list[GtHarmBenchRow]:
    rows: list[GtHarmBenchRow] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt_text = row["story_row"].strip()
            if not prompt_text:
                continue
            target_nash = row["target_nash_equilibria"].strip()
            target_util = row["target_utility_maximizing"].strip()
            target_rawls = row["target_rawlsian"].strip()
            target_nsw = row["target_nash_social_welfare"].strip()
            disagreement_bucket = classifyDisagreement(
                target_nash=target_nash,
                target_util=target_util,
                target_rawls=target_rawls,
                target_nsw=target_nsw,
            )
            rows.append(
                GtHarmBenchRow(
                    prompt_id=f"gthb_{int(row['id']):04d}",
                    source="gt_harmbench",
                    formal_game=row["formal_game"].strip(),
                    normalized_game=normalizeGameFamily(row["formal_game"]),
                    prompt_text=prompt_text,
                    story_column_text=row["story_col"].strip(),
                    actions_row=parseActionList(row["actions_row"]),
                    actions_column=parseActionList(row["actions_column"]),
                    payoff_matrix=(
                        (
                            parsePayoffCell(row["1_1_payoff"]),
                            parsePayoffCell(row["1_2_payoff"]),
                        ),
                        (
                            parsePayoffCell(row["2_1_payoff"]),
                            parsePayoffCell(row["2_2_payoff"]),
                        ),
                    ),
                    risk_level=parseOptionalInt(row["risk_level"]),
                    quality_score=parseOptionalFloat(row["quality_score"]),
                    equilibria_score=parseOptionalFloat(row["equilibria_score"]),
                    target_nash=target_nash,
                    target_util=target_util,
                    target_rawls=target_rawls,
                    target_nsw=target_nsw,
                    disagreement_bucket=disagreement_bucket,
                    disagreement_priority=DISAGREEMENT_PRIORITY[disagreement_bucket],
                )
            )
    return rows


def scoreGtHarmBenchRow(row: GtHarmBenchRow) -> tuple[float, float, float, int, str]:
    risk_score = float(row.risk_level or 0)
    quality_score = float(row.quality_score or 0.0)
    equilibria_score = float(row.equilibria_score or 0.0)
    return (
        float(row.disagreement_priority),
        risk_score,
        quality_score + equilibria_score,
        -len(row.prompt_text),
        row.prompt_id,
    )


def selectRowsForGame(
    rows: list[GtHarmBenchRow],
    normalized_game: str,
    count: int,
    excluded_prompt_ids: set[str] | None = None,
) -> list[GtHarmBenchRow]:
    blocked_prompt_ids = excluded_prompt_ids or set()
    matching_rows = [
        row
        for row in rows
        if row.normalized_game == normalized_game and row.prompt_id not in blocked_prompt_ids
    ]
    matching_rows.sort(key=scoreGtHarmBenchRow, reverse=True)
    return matching_rows[:count]


def selectDisagreementRows(
    rows: list[GtHarmBenchRow],
    count: int,
    excluded_prompt_ids: set[str] | None = None,
) -> list[GtHarmBenchRow]:
    blocked_prompt_ids = excluded_prompt_ids or set()
    candidate_rows = [row for row in rows if row.prompt_id not in blocked_prompt_ids]
    candidate_rows.sort(key=scoreGtHarmBenchRow, reverse=True)

    selected_rows: list[GtHarmBenchRow] = []
    selected_counts_by_game: dict[str, int] = {}

    for row in candidate_rows:
        if row.disagreement_priority == 0:
            continue
        game_count = selected_counts_by_game.get(row.normalized_game, 0)
        if game_count >= 12:
            continue
        selected_rows.append(row)
        selected_counts_by_game[row.normalized_game] = game_count + 1
        if len(selected_rows) == count:
            return selected_rows

    return selected_rows
