from __future__ import annotations

import ast
import csv
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prompt_schema import PromptSuiteRecord, validatePromptSuiteRecords


GAME_NORMALIZATION = {
    "prisoner's dilemma": "prisoners_dilemma",
    "stag hunt": "stag_hunt",
    "chicken": "chicken",
    "bach or stravinski": "bach_or_stravinski",
    "coordination": "coordination",
    "matching pennies": "matching_pennies",
    "no conflict": "no_conflict",
}


UNDEFINED_TARGET_MARKERS = {
    "",
    "none",
    "null",
    "nan",
    "no pure nash equilibrium",
    "no pure equilibrium",
}


@dataclass(frozen=True)
class GtHarmBenchLoadReport:
    csv_path: str
    total_rows: int
    kept_rows: int
    dropped_rows: int
    drop_reasons: dict[str, int]
    game_counts: dict[str, int]
    risk_bucket_counts: dict[str, int]
    target_agreement_counts: dict[str, int]
    ambiguity_counts: dict[str, int]


@dataclass(frozen=True)
class GtHarmBenchLoadResult:
    records: list[PromptSuiteRecord]
    report: GtHarmBenchLoadReport


class DatasetRowError(ValueError):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


GtHarmBenchRow = PromptSuiteRecord


def normalizeWhitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalizeActionLabel(value: str) -> str:
    return normalizeWhitespace(value).casefold()


def normalizeSnakeCaseLabel(value: str) -> str:
    compact_value = normalizeWhitespace(value).casefold()
    sanitized_value = re.sub(r"[^a-z0-9]+", "_", compact_value)
    sanitized_value = re.sub(r"_+", "_", sanitized_value).strip("_")
    return sanitized_value


def normalizeGameFamily(value: str) -> str:
    compact_value = normalizeWhitespace(value).casefold()
    return GAME_NORMALIZATION.get(compact_value, normalizeSnakeCaseLabel(compact_value))


def parseLiteral(value: str, field_name: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise DatasetRowError(f"invalid_{field_name}", f"{field_name} could not be parsed: {value!r}") from error


def parseOptionalInt(value: str) -> int | None:
    cleaned_value = value.strip()
    if not cleaned_value:
        return None
    return int(cleaned_value)


def parseOptionalFloat(value: str) -> float | None:
    cleaned_value = value.strip()
    if not cleaned_value:
        return None
    return float(cleaned_value)


def normalizeNumericValue(value: Any, field_name: str) -> int | float:
    if not isinstance(value, int | float):
        raise DatasetRowError(f"invalid_{field_name}", f"{field_name} must contain numeric values.")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def parseTwoActionList(raw_value: str, field_name: str) -> tuple[str, str]:
    parsed_value = parseLiteral(raw_value, field_name)
    if not isinstance(parsed_value, list | tuple) or len(parsed_value) != 2:
        raise DatasetRowError(f"invalid_{field_name}", f"{field_name} must contain exactly two actions.")

    normalized_actions = tuple(normalizeActionLabel(str(item)) for item in parsed_value)
    if any(not action for action in normalized_actions):
        raise DatasetRowError(f"invalid_{field_name}", f"{field_name} contains an empty action.")
    if normalized_actions[0] == normalized_actions[1]:
        raise DatasetRowError(f"invalid_{field_name}", f"{field_name} must contain two distinct actions.")
    return normalized_actions


def parsePayoffCell(raw_value: str, field_name: str) -> tuple[int | float, int | float]:
    parsed_value = parseLiteral(raw_value, field_name)
    if not isinstance(parsed_value, list | tuple) or len(parsed_value) != 2:
        raise DatasetRowError(f"invalid_{field_name}", f"{field_name} must contain two payoff values.")

    row_payoff = normalizeNumericValue(parsed_value[0], field_name)
    column_payoff = normalizeNumericValue(parsed_value[1], field_name)
    return row_payoff, column_payoff


def buildPayoffMatrix(
    actions_row: tuple[str, str],
    actions_column: tuple[str, str],
    cell_11: tuple[int | float, int | float],
    cell_12: tuple[int | float, int | float],
    cell_21: tuple[int | float, int | float],
    cell_22: tuple[int | float, int | float],
) -> dict[str, Any]:
    payoff_cells = (
        (cell_11, cell_12),
        (cell_21, cell_22),
    )
    matrix_rows: list[list[dict[str, Any]]] = []

    for row_index, row_action in enumerate(actions_row):
        matrix_row: list[dict[str, Any]] = []
        for column_index, column_action in enumerate(actions_column):
            row_payoff, column_payoff = payoff_cells[row_index][column_index]
            matrix_row.append(
                {
                    "row_action": row_action,
                    "column_action": column_action,
                    "row_payoff": row_payoff,
                    "column_payoff": column_payoff,
                }
            )
        matrix_rows.append(matrix_row)

    return {
        "row_actions": [actions_row[0], actions_row[1]],
        "column_actions": [actions_column[0], actions_column[1]],
        "cells": matrix_rows,
    }


def parseTargetActions(raw_value: str, field_name: str, valid_actions: tuple[str, str]) -> tuple[str, ...]:
    cleaned_value = raw_value.strip()
    if normalizeWhitespace(cleaned_value).casefold() in UNDEFINED_TARGET_MARKERS:
        return tuple()

    found_actions: set[str] = set()
    for segment in cleaned_value.split("|"):
        cleaned_segment = segment.strip()
        if not cleaned_segment:
            continue
        if normalizeWhitespace(cleaned_segment).casefold() in UNDEFINED_TARGET_MARKERS:
            continue
        parsed_segment = parseLiteral(cleaned_segment, field_name)
        if parsed_segment is None:
            continue
        if isinstance(parsed_segment, list | tuple) and parsed_segment:
            row_action = normalizeActionLabel(str(parsed_segment[0]))
        elif isinstance(parsed_segment, str):
            row_action = normalizeActionLabel(parsed_segment)
        else:
            raise DatasetRowError(f"invalid_{field_name}", f"{field_name} contains an invalid target segment.")

        if row_action not in valid_actions:
            raise DatasetRowError(
                f"invalid_{field_name}",
                f"{field_name} produced {row_action!r}, which is not a valid row-player action.",
            )
        found_actions.add(row_action)

    return tuple(action for action in valid_actions if action in found_actions)


def classifyRiskBucket(risk_level: int | None) -> str:
    if risk_level is None:
        return "missing"
    if risk_level <= 4:
        return "low"
    if risk_level <= 7:
        return "medium"
    return "high"


def classifyTargetAgreementBucket(
    target_nash: tuple[str, ...],
    target_util: tuple[str, ...],
    target_rawls: tuple[str, ...],
    target_nsw: tuple[str, ...],
) -> str:
    normative_values = {target_util, target_rawls, target_nsw}
    all_values = {target_nash, target_util, target_rawls, target_nsw}

    if not target_nash:
        return "nash_undefined" if len(normative_values) == 1 else "normative_split"
    if len(all_values) == 1:
        return "all_agree"
    if len(normative_values) == 1 and target_nash != target_util:
        return "nash_vs_normative"
    if len(normative_values) > 1:
        return "normative_split"
    return "mixed"


def classifyAmbiguityBucket(
    target_nash: tuple[str, ...],
    target_util: tuple[str, ...],
    target_rawls: tuple[str, ...],
    target_nsw: tuple[str, ...],
) -> str:
    targets = (target_nash, target_util, target_rawls, target_nsw)
    if any(len(target) != 1 for target in targets):
        return "strategically_ambiguous"
    return "strategically_easy"


def classifyTargetActionType(
    actions: tuple[str, str],
    target_nash: tuple[str, ...],
    target_util: tuple[str, ...],
    target_rawls: tuple[str, ...],
    target_nsw: tuple[str, ...],
) -> str:
    observed_actions = {
        action
        for target_values in (target_nash, target_util, target_rawls, target_nsw)
        for action in target_values
    }
    if not observed_actions:
        return "no_defined_target"
    if len(observed_actions) == 1:
        return "action_0_only" if actions[0] in observed_actions else "action_1_only"
    return "mixed_actions"


def buildPromptId(source_row_id: int) -> str:
    return f"gthb_{source_row_id:04d}"


def buildPromptSuiteRecordFromCsvRow(row: dict[str, str]) -> PromptSuiteRecord:
    raw_prompt_text = row.get("story_row", "")
    prompt_text = raw_prompt_text.strip()
    if not prompt_text:
        raise DatasetRowError("missing_story_row", "story_row is empty.")

    raw_formal_game = row.get("formal_game", "")
    formal_game_raw = raw_formal_game.strip()
    if not formal_game_raw:
        raise DatasetRowError("missing_formal_game", "formal_game is empty.")

    raw_source_row_id = str(row.get("id", "")).strip()
    if not raw_source_row_id:
        raise DatasetRowError("missing_id", "id is empty.")

    try:
        source_row_id = int(raw_source_row_id)
    except ValueError as error:
        raise DatasetRowError("invalid_id", f"id is not an integer: {raw_source_row_id!r}") from error

    actions = parseTwoActionList(row.get("actions_row", ""), "actions_row")
    actions_column = parseTwoActionList(row.get("actions_column", ""), "actions_column")

    cell_11 = parsePayoffCell(row.get("1_1_payoff", ""), "1_1_payoff")
    cell_12 = parsePayoffCell(row.get("1_2_payoff", ""), "1_2_payoff")
    cell_21 = parsePayoffCell(row.get("2_1_payoff", ""), "2_1_payoff")
    cell_22 = parsePayoffCell(row.get("2_2_payoff", ""), "2_2_payoff")

    payoff_matrix = buildPayoffMatrix(actions, actions_column, cell_11, cell_12, cell_21, cell_22)
    target_nash_raw = row.get("target_nash_equilibria", "").strip()
    target_util_raw = row.get("target_utility_maximizing", "").strip()
    target_rawls_raw = row.get("target_rawlsian", "").strip()
    target_nsw_raw = row.get("target_nash_social_welfare", "").strip()

    target_nash = parseTargetActions(target_nash_raw, "target_nash_equilibria", actions)
    target_util = parseTargetActions(target_util_raw, "target_utility_maximizing", actions)
    target_rawls = parseTargetActions(target_rawls_raw, "target_rawlsian", actions)
    target_nsw = parseTargetActions(target_nsw_raw, "target_nash_social_welfare", actions)

    risk_level = parseOptionalInt(row.get("risk_level", ""))
    quality_score = parseOptionalFloat(row.get("quality_score", ""))
    equilibria_score = parseOptionalFloat(row.get("equilibria_score", ""))

    target_agreement_bucket = classifyTargetAgreementBucket(target_nash, target_util, target_rawls, target_nsw)
    ambiguity_bucket = classifyAmbiguityBucket(target_nash, target_util, target_rawls, target_nsw)

    return PromptSuiteRecord(
        prompt_id=buildPromptId(source_row_id),
        source_row_id=source_row_id,
        source="gt_harmbench",
        subset="master_pool",
        probe_split="not_applicable",
        formal_game=normalizeGameFamily(formal_game_raw),
        formal_game_raw=formal_game_raw,
        prompt_text=prompt_text,
        actions=actions,
        num_actions=len(actions),
        actions_raw=row.get("actions_row", ""),
        payoff_matrix=payoff_matrix,
        payoff_matrix_raw={
            "actions_row": row.get("actions_row", ""),
            "actions_column": row.get("actions_column", ""),
            "1_1_payoff": row.get("1_1_payoff", ""),
            "1_2_payoff": row.get("1_2_payoff", ""),
            "2_1_payoff": row.get("2_1_payoff", ""),
            "2_2_payoff": row.get("2_2_payoff", ""),
        },
        risk_level=risk_level,
        risk_bucket=classifyRiskBucket(risk_level),
        story_side="row",
        payoff_table_present=True,
        history_type="one_shot",
        target_nash=target_nash,
        target_util=target_util,
        target_rawls=target_rawls,
        target_nsw=target_nsw,
        target_nash_raw=target_nash_raw,
        target_util_raw=target_util_raw,
        target_rawls_raw=target_rawls_raw,
        target_nsw_raw=target_nsw_raw,
        target_agreement_bucket=target_agreement_bucket,
        ambiguity_bucket=ambiguity_bucket,
        target_action_type=classifyTargetActionType(actions, target_nash, target_util, target_rawls, target_nsw),
        quality_score=quality_score,
        equilibria_score=equilibria_score,
    )


def buildLoadReport(csv_path: Path, records: list[PromptSuiteRecord], drop_reasons: Counter[str], total_rows: int) -> GtHarmBenchLoadReport:
    game_counts = Counter(record.formal_game for record in records)
    risk_bucket_counts = Counter(record.risk_bucket for record in records)
    target_agreement_counts = Counter(record.target_agreement_bucket for record in records)
    ambiguity_counts = Counter(record.ambiguity_bucket for record in records)

    return GtHarmBenchLoadReport(
        csv_path=str(csv_path),
        total_rows=total_rows,
        kept_rows=len(records),
        dropped_rows=total_rows - len(records),
        drop_reasons=dict(sorted(drop_reasons.items())),
        game_counts=dict(sorted(game_counts.items())),
        risk_bucket_counts=dict(sorted(risk_bucket_counts.items())),
        target_agreement_counts=dict(sorted(target_agreement_counts.items())),
        ambiguity_counts=dict(sorted(ambiguity_counts.items())),
    )


def loadGtHarmBenchDataset(csv_path: Path, logger: logging.Logger | None = None) -> GtHarmBenchLoadResult:
    records: list[PromptSuiteRecord] = []
    drop_reasons: Counter[str] = Counter()
    total_rows = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1
            try:
                records.append(buildPromptSuiteRecordFromCsvRow(row))
            except DatasetRowError as error:
                drop_reasons[error.reason] += 1
                if logger is not None:
                    logger.warning("Dropping GT-HarmBench row id=%s: %s", row.get("id", "?"), error)

    validatePromptSuiteRecords(records)
    report = buildLoadReport(csv_path, records, drop_reasons, total_rows)

    if logger is not None:
        logger.info(
            "Loaded GT-HarmBench master pool: kept=%s dropped=%s total=%s",
            report.kept_rows,
            report.dropped_rows,
            report.total_rows,
        )

    return GtHarmBenchLoadResult(records=records, report=report)


def loadGtHarmBenchRows(csv_path: Path) -> list[GtHarmBenchRow]:
    return loadGtHarmBenchDataset(csv_path).records
