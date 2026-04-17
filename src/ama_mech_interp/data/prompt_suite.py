from __future__ import annotations

from collections import Counter
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from ..storage import writeJsonlFile
from .gt_harmbench import GtHarmBenchRow, loadGtHarmBenchRows
from .prompt_schema import PromptSuiteBuildConfig, PromptSuiteRecord, validatePromptSuiteRecords


PromptSuiteRow = PromptSuiteRecord


GAME_PRIORITY_ORDER = (
    "prisoners_dilemma",
    "stag_hunt",
    "chicken",
    "bach_or_stravinski",
    "coordination",
    "matching_pennies",
    "no_conflict",
)


DISAGREEMENT_GAME_WEIGHTS = {
    "chicken": 3.0,
    "prisoners_dilemma": 2.0,
    "stag_hunt": 2.0,
    "matching_pennies": 1.0,
    "bach_or_stravinski": 1.0,
}


MINIMUM_VIABLE_GAME_QUOTAS = {
    "prisoners_dilemma": 8,
    "stag_hunt": 8,
    "chicken": 8,
    "bach_or_stravinski": 8,
    "coordination": 8,
}


SUBSET_SORT_ORDER = {
    "analysis": 0,
    "disagreement": 1,
    "heldout_replication": 2,
    "master_pool": 3,
}


PROBE_SPLIT_SORT_ORDER = {
    "train": 0,
    "validation": 1,
    "test": 2,
    "not_applicable": 3,
}


def buildSelectionStratum(record: PromptSuiteRecord) -> tuple[str, str, str, str]:
    return (
        record.risk_bucket,
        record.target_agreement_bucket,
        record.ambiguity_bucket,
        record.target_action_type,
    )


def buildGeneralRankingKey(record: PromptSuiteRecord) -> tuple[float, float, int, str]:
    return (
        -(record.quality_score or 0.0),
        -(record.equilibria_score or 0.0),
        -(record.risk_level or 0),
        record.prompt_id,
    )


def buildDisagreementRankingKey(record: PromptSuiteRecord) -> tuple[float, int, float, float, str]:
    disagreement_priority = {
        "normative_split": 4.0,
        "nash_vs_normative": 3.0,
        "nash_undefined": 2.0,
        "mixed": 1.0,
        "all_agree": 0.0,
    }[record.target_agreement_bucket]
    ambiguity_priority = 1 if record.ambiguity_bucket == "strategically_ambiguous" else 0
    return (
        -disagreement_priority,
        -ambiguity_priority,
        -(record.quality_score or 0.0),
        -(record.equilibria_score or 0.0),
        record.prompt_id,
    )


def takeBalancedSample(
    records: list[PromptSuiteRecord],
    count: int,
    ranking_builder,
) -> list[PromptSuiteRecord]:
    grouped_records: dict[tuple[str, str, str, str], list[PromptSuiteRecord]] = defaultdict(list)
    for record in records:
        grouped_records[buildSelectionStratum(record)].append(record)

    for stratum_key, stratum_records in grouped_records.items():
        grouped_records[stratum_key] = sorted(stratum_records, key=ranking_builder)

    selected_records: list[PromptSuiteRecord] = []

    while len(selected_records) < count and grouped_records:
        ordered_keys = sorted(grouped_records, key=lambda key: (-len(grouped_records[key]), str(key)))
        progress_made = False

        for stratum_key in ordered_keys:
            if not grouped_records[stratum_key]:
                continue
            selected_records.append(grouped_records[stratum_key].pop(0))
            progress_made = True
            if len(selected_records) == count:
                break

        grouped_records = {key: value for key, value in grouped_records.items() if value}
        if not progress_made:
            break

    return selected_records


def allocateEvenGameQuotas(records: list[PromptSuiteRecord], total_count: int) -> dict[str, int]:
    available_counts = {game: sum(1 for record in records if record.formal_game == game) for game in GAME_PRIORITY_ORDER}
    quotas = {game: 0 for game in GAME_PRIORITY_ORDER}
    remaining = total_count

    while remaining > 0:
        progress_made = False
        for game in GAME_PRIORITY_ORDER:
            if available_counts[game] <= quotas[game]:
                continue
            quotas[game] += 1
            remaining -= 1
            progress_made = True
            if remaining == 0:
                break
        if not progress_made:
            break

    if remaining > 0:
        raise ValueError(f"Unable to allocate {total_count} prompts evenly across games.")

    return {game: count for game, count in quotas.items() if count > 0}


def allocateWeightedGameQuotas(records: list[PromptSuiteRecord], total_count: int) -> dict[str, int]:
    available_counts = {game: sum(1 for record in records if record.formal_game == game) for game in DISAGREEMENT_GAME_WEIGHTS}
    quotas = {game: 0 for game in DISAGREEMENT_GAME_WEIGHTS}
    remaining = total_count

    while remaining > 0:
        candidate_games = [
            game
            for game in DISAGREEMENT_GAME_WEIGHTS
            if quotas[game] < available_counts[game]
        ]
        if not candidate_games:
            break

        next_game = min(
            candidate_games,
            key=lambda game: (quotas[game] / DISAGREEMENT_GAME_WEIGHTS[game], GAME_PRIORITY_ORDER.index(game)),
        )
        quotas[next_game] += 1
        remaining -= 1

    if remaining > 0:
        raise ValueError(f"Unable to allocate {total_count} prompts for the disagreement pool.")

    return {game: count for game, count in quotas.items() if count > 0}


def selectRecordsByGameQuotas(
    records: list[PromptSuiteRecord],
    game_quotas: dict[str, int],
    ranking_builder,
) -> list[PromptSuiteRecord]:
    selected_records: list[PromptSuiteRecord] = []
    for game in GAME_PRIORITY_ORDER:
        quota = game_quotas.get(game, 0)
        if quota == 0:
            continue
        game_records = [record for record in records if record.formal_game == game]
        chosen_records = takeBalancedSample(game_records, quota, ranking_builder)
        if len(chosen_records) != quota:
            raise ValueError(f"Unable to select {quota} prompts for game {game}.")
        selected_records.extend(chosen_records)
    return selected_records


def applySubsetMetadata(
    records: list[PromptSuiteRecord],
    subset: str,
    probe_split: str = "not_applicable",
) -> list[PromptSuiteRecord]:
    return [replace(record, subset=subset, probe_split=probe_split) for record in records]


def buildProbeSplitCounts(record_count: int) -> dict[str, int]:
    train_count = record_count // 2
    remaining = record_count - train_count
    validation_count = remaining // 2
    test_count = record_count - train_count - validation_count
    return {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
    }


def assignProbeSplits(analysis_records: list[PromptSuiteRecord]) -> list[PromptSuiteRecord]:
    records_by_game: dict[str, list[PromptSuiteRecord]] = defaultdict(list)
    for record in analysis_records:
        records_by_game[record.formal_game].append(record)

    split_records: list[PromptSuiteRecord] = []
    for game in GAME_PRIORITY_ORDER:
        game_records = records_by_game.get(game, [])
        if not game_records:
            continue

        split_counts = buildProbeSplitCounts(len(game_records))
        remaining_records = sorted(game_records, key=buildGeneralRankingKey)

        train_records = takeBalancedSample(remaining_records, split_counts["train"], buildGeneralRankingKey)
        remaining_records = [record for record in remaining_records if record.prompt_id not in {item.prompt_id for item in train_records}]

        validation_records = takeBalancedSample(remaining_records, split_counts["validation"], buildGeneralRankingKey)
        remaining_records = [record for record in remaining_records if record.prompt_id not in {item.prompt_id for item in validation_records}]

        test_records = takeBalancedSample(remaining_records, split_counts["test"], buildGeneralRankingKey)

        split_records.extend(applySubsetMetadata(train_records, subset="analysis", probe_split="train"))
        split_records.extend(applySubsetMetadata(validation_records, subset="analysis", probe_split="validation"))
        split_records.extend(applySubsetMetadata(test_records, subset="analysis", probe_split="test"))

    split_records.sort(key=lambda record: (record.formal_game, PROBE_SPLIT_SORT_ORDER[record.probe_split], record.prompt_id))
    return split_records


def sortControlledPromptSuite(records: list[PromptSuiteRecord]) -> list[PromptSuiteRecord]:
    return sorted(
        records,
        key=lambda record: (
            SUBSET_SORT_ORDER[record.subset],
            record.formal_game,
            PROBE_SPLIT_SORT_ORDER[record.probe_split],
            record.prompt_id,
        ),
    )


def buildControlledPromptSuite(
    master_records: list[PromptSuiteRecord],
    config: PromptSuiteBuildConfig | None = None,
) -> list[PromptSuiteRecord]:
    build_config = config or PromptSuiteBuildConfig()

    analysis_game_quotas = allocateEvenGameQuotas(master_records, build_config.analysis_pool_count)
    analysis_seed_records = selectRecordsByGameQuotas(master_records, analysis_game_quotas, buildGeneralRankingKey)
    analysis_prompt_ids = {record.prompt_id for record in analysis_seed_records}

    remaining_after_analysis = [record for record in master_records if record.prompt_id not in analysis_prompt_ids]

    disagreement_candidates = [
        record
        for record in remaining_after_analysis
        if record.formal_game in DISAGREEMENT_GAME_WEIGHTS
        and (
            record.target_agreement_bucket != "all_agree"
            or record.ambiguity_bucket == "strategically_ambiguous"
        )
    ]
    disagreement_game_quotas = allocateWeightedGameQuotas(disagreement_candidates, build_config.disagreement_pool_count)
    disagreement_records = applySubsetMetadata(
        selectRecordsByGameQuotas(disagreement_candidates, disagreement_game_quotas, buildDisagreementRankingKey),
        subset="disagreement",
    )
    disagreement_prompt_ids = {record.prompt_id for record in disagreement_records}

    remaining_after_disagreement = [
        record
        for record in remaining_after_analysis
        if record.prompt_id not in disagreement_prompt_ids
    ]
    heldout_game_quotas = allocateEvenGameQuotas(remaining_after_disagreement, build_config.heldout_pool_count)
    heldout_records = applySubsetMetadata(
        selectRecordsByGameQuotas(remaining_after_disagreement, heldout_game_quotas, buildGeneralRankingKey),
        subset="heldout_replication",
    )

    analysis_records = assignProbeSplits(analysis_seed_records)
    controlled_suite = sortControlledPromptSuite([*analysis_records, *disagreement_records, *heldout_records])

    probe_split_counts = Counter(record.probe_split for record in controlled_suite if record.subset == "analysis")
    if probe_split_counts["train"] != build_config.probe_train_count:
        raise ValueError("Analysis probe train split does not match configured count.")
    if probe_split_counts["validation"] != build_config.probe_validation_count:
        raise ValueError("Analysis probe validation split does not match configured count.")
    if probe_split_counts["test"] != build_config.probe_test_count:
        raise ValueError("Analysis probe test split does not match configured count.")
    if len(controlled_suite) != (
        build_config.analysis_pool_count
        + build_config.disagreement_pool_count
        + build_config.heldout_pool_count
    ):
        raise ValueError("Controlled suite count does not match configured totals.")

    validatePromptSuiteRecords(controlled_suite)
    return controlled_suite


def buildDefaultPromptSuite(csv_path: Path) -> list[PromptSuiteRow]:
    return buildControlledPromptSuite(loadGtHarmBenchRows(csv_path))


def buildMinimumViablePromptSuite(prompt_suite_rows: list[PromptSuiteRow]) -> list[PromptSuiteRow]:
    analysis_rows = [row for row in prompt_suite_rows if row.subset == "analysis"]
    selected_rows: list[PromptSuiteRow] = []

    for game in GAME_PRIORITY_ORDER:
        quota = MINIMUM_VIABLE_GAME_QUOTAS.get(game, 0)
        if quota == 0:
            continue
        game_rows = [row for row in analysis_rows if row.formal_game == game]
        game_rows.sort(key=lambda row: (PROBE_SPLIT_SORT_ORDER[row.probe_split], row.prompt_id))
        selected_rows.extend(game_rows[:quota])

    return sortControlledPromptSuite(selected_rows)


def writePromptSuite(path: Path, prompt_suite_rows: list[PromptSuiteRow]) -> Path:
    validatePromptSuiteRecords(prompt_suite_rows)
    return writeJsonlFile(path, prompt_suite_rows)
