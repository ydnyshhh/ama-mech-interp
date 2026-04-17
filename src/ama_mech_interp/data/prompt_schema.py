from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any


@dataclass(frozen=True)
class PromptSuiteRecord:
    prompt_id: str
    source_row_id: int
    source: str
    subset: str
    probe_split: str
    formal_game: str
    formal_game_raw: str
    prompt_text: str
    actions: tuple[str, str]
    num_actions: int
    actions_raw: str
    payoff_matrix: dict[str, Any]
    payoff_matrix_raw: dict[str, str]
    risk_level: int | None
    risk_bucket: str
    story_side: str
    payoff_table_present: bool
    history_type: str
    target_nash: tuple[str, ...]
    target_util: tuple[str, ...]
    target_rawls: tuple[str, ...]
    target_nsw: tuple[str, ...]
    target_nash_raw: str
    target_util_raw: str
    target_rawls_raw: str
    target_nsw_raw: str
    target_agreement_bucket: str
    ambiguity_bucket: str
    target_action_type: str
    quality_score: float | None
    equilibria_score: float | None


@dataclass(frozen=True)
class PromptSuiteBuildConfig:
    analysis_pool_count: int = 140
    disagreement_pool_count: int = 30
    heldout_pool_count: int = 30
    probe_train_count: int = 70
    probe_validation_count: int = 35
    probe_test_count: int = 35


class PromptSchemaError(ValueError):
    pass


def validateTargetList(value: tuple[str, ...], field_name: str) -> None:
    if not isinstance(value, tuple):
        raise PromptSchemaError(f"{field_name} must be a tuple.")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise PromptSchemaError(f"{field_name} contains an invalid action value.")


def validateTargetPayload(value: Any, field_name: str) -> None:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise PromptSchemaError(f"{field_name} must be a list in the serialized artifact.")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise PromptSchemaError(f"{field_name} contains an invalid action value.")


def validatePayoffMatrix(payoff_matrix: dict[str, Any]) -> None:
    if not isinstance(payoff_matrix, dict):
        raise PromptSchemaError("payoff_matrix must be a dictionary.")

    row_actions = payoff_matrix.get("row_actions")
    column_actions = payoff_matrix.get("column_actions")
    cells = payoff_matrix.get("cells")

    if not isinstance(row_actions, list) or len(row_actions) != 2:
        raise PromptSchemaError("payoff_matrix.row_actions must be a two-item list.")
    if not isinstance(column_actions, list) or len(column_actions) != 2:
        raise PromptSchemaError("payoff_matrix.column_actions must be a two-item list.")
    if not isinstance(cells, list) or len(cells) != 2:
        raise PromptSchemaError("payoff_matrix.cells must be a 2x2 list.")

    for row in cells:
        if not isinstance(row, list) or len(row) != 2:
            raise PromptSchemaError("payoff_matrix.cells must be a 2x2 list.")
        for cell in row:
            if not isinstance(cell, dict):
                raise PromptSchemaError("Each payoff cell must be a dictionary.")
            row_payoff = cell.get("row_payoff")
            column_payoff = cell.get("column_payoff")
            if not isinstance(row_payoff, Real):
                raise PromptSchemaError("row_payoff must be numeric.")
            if not isinstance(column_payoff, Real):
                raise PromptSchemaError("column_payoff must be numeric.")


def validatePromptSuiteRecord(record: PromptSuiteRecord) -> None:
    if not record.prompt_id.strip():
        raise PromptSchemaError("prompt_id must be non-empty.")
    if not record.prompt_text.strip():
        raise PromptSchemaError(f"{record.prompt_id}: prompt_text must be non-empty.")
    if len(record.actions) != 2:
        raise PromptSchemaError(f"{record.prompt_id}: actions must contain exactly two actions.")
    if record.num_actions != len(record.actions):
        raise PromptSchemaError(f"{record.prompt_id}: num_actions does not match actions length.")
    for action in record.actions:
        if not action.strip():
            raise PromptSchemaError(f"{record.prompt_id}: actions contains an empty action.")
    if record.story_side != "row":
        raise PromptSchemaError(f"{record.prompt_id}: story_side must be 'row'.")
    validatePayoffMatrix(record.payoff_matrix)
    validateTargetList(record.target_nash, "target_nash")
    validateTargetList(record.target_util, "target_util")
    validateTargetList(record.target_rawls, "target_rawls")
    validateTargetList(record.target_nsw, "target_nsw")


def validatePromptSuiteRecords(records: list[PromptSuiteRecord]) -> None:
    seen_prompt_ids: set[str] = set()
    duplicate_prompt_ids: set[str] = set()

    for record in records:
        validatePromptSuiteRecord(record)
        if record.prompt_id in seen_prompt_ids:
            duplicate_prompt_ids.add(record.prompt_id)
        seen_prompt_ids.add(record.prompt_id)

    if duplicate_prompt_ids:
        duplicate_list = ", ".join(sorted(duplicate_prompt_ids))
        raise PromptSchemaError(f"Duplicate prompt ids found: {duplicate_list}")


def validatePromptSuitePayload(record: Mapping[str, Any]) -> None:
    prompt_id = str(record.get("prompt_id", "")).strip()
    if not prompt_id:
        raise PromptSchemaError("Serialized prompt record is missing prompt_id.")
    prompt_text = str(record.get("prompt_text", "")).strip()
    if not prompt_text:
        raise PromptSchemaError(f"{prompt_id}: prompt_text must be non-empty.")

    actions = record.get("actions")
    if not isinstance(actions, list) or len(actions) != 2:
        raise PromptSchemaError(f"{prompt_id}: serialized actions must be a two-item list.")
    for action in actions:
        if not isinstance(action, str) or not action.strip():
            raise PromptSchemaError(f"{prompt_id}: serialized actions contains an invalid value.")

    num_actions = record.get("num_actions")
    if num_actions != len(actions):
        raise PromptSchemaError(f"{prompt_id}: num_actions does not match serialized actions length.")

    story_side = str(record.get("story_side", ""))
    if story_side != "row":
        raise PromptSchemaError(f"{prompt_id}: story_side must be 'row'.")

    payoff_matrix = record.get("payoff_matrix")
    if not isinstance(payoff_matrix, dict):
        raise PromptSchemaError(f"{prompt_id}: serialized payoff_matrix must be a dictionary.")
    validatePayoffMatrix(payoff_matrix)

    validateTargetPayload(record.get("target_nash"), "target_nash")
    validateTargetPayload(record.get("target_util"), "target_util")
    validateTargetPayload(record.get("target_rawls"), "target_rawls")
    validateTargetPayload(record.get("target_nsw"), "target_nsw")


def validatePromptSuitePayloadRows(records: list[dict[str, Any]]) -> None:
    seen_prompt_ids: set[str] = set()
    duplicate_prompt_ids: set[str] = set()

    for record in records:
        validatePromptSuitePayload(record)
        prompt_id = str(record["prompt_id"])
        if prompt_id in seen_prompt_ids:
            duplicate_prompt_ids.add(prompt_id)
        seen_prompt_ids.add(prompt_id)

    if duplicate_prompt_ids:
        duplicate_list = ", ".join(sorted(duplicate_prompt_ids))
        raise PromptSchemaError(f"Duplicate serialized prompt ids found: {duplicate_list}")
