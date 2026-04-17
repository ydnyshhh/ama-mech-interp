from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .data.gt_harmbench import loadGtHarmBenchRows
from .data.prompt_schema import PromptSuiteRecord


PromptRecord = PromptSuiteRecord


@dataclass(frozen=True)
class DatasetSummary:
    row_count: int
    missing_story_count: int
    game_counts: dict[str, int]
    risk_counts: dict[str, int]
    disagreement_counts: dict[str, int]


def loadPromptRecords(csv_path: Path) -> list[PromptRecord]:
    return loadGtHarmBenchRows(csv_path)


def summarizeDataset(records: list[PromptRecord]) -> DatasetSummary:
    game_counts = Counter(record.formal_game for record in records)
    risk_counts = Counter(str(record.risk_level) if record.risk_level is not None else "missing" for record in records)
    disagreement_counts = Counter(record.target_agreement_bucket for record in records)
    return DatasetSummary(
        row_count=len(records),
        missing_story_count=0,
        game_counts=dict(game_counts.most_common()),
        risk_counts=dict(risk_counts.most_common()),
        disagreement_counts=dict(disagreement_counts.most_common()),
    )
