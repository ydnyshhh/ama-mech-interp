from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..storage import convertToJsonReady, readJsonlFile, writeJsonFile, writeJsonlFile
from .gt_harmbench import GtHarmBenchLoadResult, loadGtHarmBenchDataset
from .prompt_schema import (
    PromptSuiteBuildConfig,
    PromptSuiteRecord,
    validatePromptSuitePayloadRows,
    validatePromptSuiteRecords,
)
from .prompt_suite import buildControlledPromptSuite


DEFAULT_DATASET_PATH = Path("gt-harmbench-dataset/gt-harmbench-with-targets.csv")
DEFAULT_ARTIFACTS_ROOT = Path("artifacts")
DEFAULT_MASTER_POOL_PATH = DEFAULT_ARTIFACTS_ROOT / "prompt_pool_master_v1.jsonl"
DEFAULT_SUITE_PATH = DEFAULT_ARTIFACTS_ROOT / "prompt_suite_v1.jsonl"
DEFAULT_MASTER_POOL_CSV_PATH = DEFAULT_ARTIFACTS_ROOT / "prompt_pool_master_v1.csv"
DEFAULT_SUITE_CSV_PATH = DEFAULT_ARTIFACTS_ROOT / "prompt_suite_v1.csv"
DEFAULT_REPORT_PATH = DEFAULT_ARTIFACTS_ROOT / "prompt_suite_v1_build_report.json"
DEFAULT_MANIFEST_PATH = DEFAULT_ARTIFACTS_ROOT / "prompt_suite_manifest.json"


CSV_FIELD_ORDER = (
    "prompt_id",
    "source_row_id",
    "source",
    "subset",
    "probe_split",
    "formal_game",
    "formal_game_raw",
    "prompt_text",
    "actions",
    "num_actions",
    "actions_raw",
    "payoff_matrix",
    "payoff_matrix_raw",
    "risk_level",
    "risk_bucket",
    "story_side",
    "payoff_table_present",
    "history_type",
    "target_nash",
    "target_util",
    "target_rawls",
    "target_nsw",
    "target_nash_raw",
    "target_util_raw",
    "target_rawls_raw",
    "target_nsw_raw",
    "target_agreement_bucket",
    "ambiguity_bucket",
    "target_action_type",
    "quality_score",
    "equilibria_score",
)


@dataclass(frozen=True)
class PromptSuiteArtifactBundle:
    master_pool_path: Path
    suite_path: Path
    master_pool_csv_path: Path | None
    suite_csv_path: Path | None
    report_path: Path
    manifest_path: Path
    master_pool_count: int
    suite_count: int


def buildRecordsSummary(records: list[PromptSuiteRecord]) -> dict[str, Any]:
    subset_counts = Counter(record.subset for record in records)
    game_counts = Counter(record.formal_game for record in records)
    risk_bucket_counts = Counter(record.risk_bucket for record in records)
    target_agreement_counts = Counter(record.target_agreement_bucket for record in records)
    ambiguity_counts = Counter(record.ambiguity_bucket for record in records)
    probe_split_counts = Counter(record.probe_split for record in records)

    per_subset_game_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        per_subset_game_counts[record.subset][record.formal_game] += 1

    return {
        "row_count": len(records),
        "subset_counts": dict(sorted(subset_counts.items())),
        "game_counts": dict(sorted(game_counts.items())),
        "risk_bucket_counts": dict(sorted(risk_bucket_counts.items())),
        "target_agreement_counts": dict(sorted(target_agreement_counts.items())),
        "ambiguity_counts": dict(sorted(ambiguity_counts.items())),
        "probe_split_counts": dict(sorted(probe_split_counts.items())),
        "per_subset_game_counts": {
            subset: dict(sorted(game_counts.items()))
            for subset, game_counts in sorted(per_subset_game_counts.items())
        },
    }


def buildBuildReport(
    csv_path: Path,
    load_result: GtHarmBenchLoadResult,
    master_records: list[PromptSuiteRecord],
    controlled_suite_records: list[PromptSuiteRecord],
    config: PromptSuiteBuildConfig,
    bundle: PromptSuiteArtifactBundle,
) -> dict[str, Any]:
    return {
        "csv_path": str(csv_path),
        "notes": {
            "canonical_story_side": "Phase one keeps only the row-player perspective and uses story_row as prompt_text.",
            "controlled_suite_split": (
                "The default controlled suite uses an internally consistent 140 analysis prompts, "
                "30 disagreement prompts, and 30 held-out replication prompts for 200 total."
            ),
            "disagreement_pool_policy": (
                "The disagreement pool is pre-behavior and is proxy-curated from target disagreement, "
                "strategic ambiguity, and prompt quality. Replace it with observed model disagreement once behavior runs exist."
            ),
        },
        "build_config": convertToJsonReady(config),
        "load_report": convertToJsonReady(load_result.report),
        "master_pool": {
            "path": str(bundle.master_pool_path),
            "csv_path": str(bundle.master_pool_csv_path) if bundle.master_pool_csv_path is not None else None,
            "summary": buildRecordsSummary(master_records),
        },
        "controlled_suite": {
            "path": str(bundle.suite_path),
            "csv_path": str(bundle.suite_csv_path) if bundle.suite_csv_path is not None else None,
            "summary": buildRecordsSummary(controlled_suite_records),
        },
        "validation": {
            "master_records": "passed",
            "controlled_suite_records": "passed",
            "master_jsonl_payload": "passed",
            "controlled_suite_jsonl_payload": "passed",
        },
    }


def buildCsvRow(record: PromptSuiteRecord) -> dict[str, Any]:
    json_ready_record = convertToJsonReady(record)
    csv_row: dict[str, Any] = {}

    for field_name in CSV_FIELD_ORDER:
        value = json_ready_record[field_name]
        if isinstance(value, list | dict):
            csv_row[field_name] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            csv_row[field_name] = value

    return csv_row


def writeCompanionCsv(path: Path, records: list[PromptSuiteRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELD_ORDER))
        writer.writeheader()
        for record in records:
            writer.writerow(buildCsvRow(record))
    return path


def materializePromptSuiteArtifacts(
    csv_path: Path,
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT,
    config: PromptSuiteBuildConfig | None = None,
    write_companion_csv: bool = True,
    logger: logging.Logger | None = None,
) -> PromptSuiteArtifactBundle:
    build_config = config or PromptSuiteBuildConfig()
    active_logger = logger or logging.getLogger(__name__)

    load_result = loadGtHarmBenchDataset(csv_path, logger=active_logger)
    master_records = load_result.records
    controlled_suite_records = buildControlledPromptSuite(master_records, build_config)

    validatePromptSuiteRecords(master_records)
    validatePromptSuiteRecords(controlled_suite_records)

    master_pool_path = artifacts_root / DEFAULT_MASTER_POOL_PATH.name
    suite_path = artifacts_root / DEFAULT_SUITE_PATH.name
    master_pool_csv_path = artifacts_root / DEFAULT_MASTER_POOL_CSV_PATH.name if write_companion_csv else None
    suite_csv_path = artifacts_root / DEFAULT_SUITE_CSV_PATH.name if write_companion_csv else None
    report_path = artifacts_root / DEFAULT_REPORT_PATH.name
    manifest_path = artifacts_root / DEFAULT_MANIFEST_PATH.name

    writeJsonlFile(master_pool_path, master_records)
    writeJsonlFile(suite_path, controlled_suite_records)

    if write_companion_csv:
        writeCompanionCsv(master_pool_csv_path, master_records)
        writeCompanionCsv(suite_csv_path, controlled_suite_records)

    validatePromptSuitePayloadRows(readJsonlFile(master_pool_path))
    validatePromptSuitePayloadRows(readJsonlFile(suite_path))

    bundle = PromptSuiteArtifactBundle(
        master_pool_path=master_pool_path,
        suite_path=suite_path,
        master_pool_csv_path=master_pool_csv_path,
        suite_csv_path=suite_csv_path,
        report_path=report_path,
        manifest_path=manifest_path,
        master_pool_count=len(master_records),
        suite_count=len(controlled_suite_records),
    )

    build_report = buildBuildReport(csv_path, load_result, master_records, controlled_suite_records, build_config, bundle)
    writeJsonFile(report_path, build_report)
    writeJsonFile(
        manifest_path,
        {
            "master_pool_path": str(master_pool_path),
            "suite_path": str(suite_path),
            "master_pool_count": len(master_records),
            "suite_count": len(controlled_suite_records),
            "report_path": str(report_path),
        },
    )

    active_logger.info("Wrote master pool to %s (%s rows)", master_pool_path, len(master_records))
    active_logger.info("Wrote controlled suite to %s (%s rows)", suite_path, len(controlled_suite_records))

    return bundle


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the canonical GT-HarmBench prompt-suite artifacts.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATASET_PATH, help="Path to gt-harmbench-with-targets.csv.")
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT, help="Directory for output artifacts.")
    parser.add_argument("--skip-csv", action="store_true", help="Do not write companion CSV artifacts.")
    parser.add_argument("--analysis-pool-count", type=int, default=PromptSuiteBuildConfig.analysis_pool_count, help="Controlled-suite analysis pool size.")
    parser.add_argument("--disagreement-pool-count", type=int, default=PromptSuiteBuildConfig.disagreement_pool_count, help="Controlled-suite disagreement pool size.")
    parser.add_argument("--heldout-pool-count", type=int, default=PromptSuiteBuildConfig.heldout_pool_count, help="Controlled-suite held-out replication pool size.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = buildParser()
    args = parser.parse_args()

    config = PromptSuiteBuildConfig(
        analysis_pool_count=args.analysis_pool_count,
        disagreement_pool_count=args.disagreement_pool_count,
        heldout_pool_count=args.heldout_pool_count,
    )
    bundle = materializePromptSuiteArtifacts(
        csv_path=args.csv,
        artifacts_root=args.artifacts_root,
        config=config,
        write_companion_csv=not args.skip_csv,
        logger=logging.getLogger("ama_mech_interp.build_prompt_suite"),
    )

    print(f"master_pool_path: {bundle.master_pool_path}")
    print(f"master_pool_count: {bundle.master_pool_count}")
    print(f"suite_path: {bundle.suite_path}")
    print(f"suite_count: {bundle.suite_count}")
    print(f"report_path: {bundle.report_path}")


if __name__ == "__main__":
    main()
