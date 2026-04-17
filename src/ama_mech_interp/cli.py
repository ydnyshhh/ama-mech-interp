from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import loadPromptRecords, summarizeDataset
from .models import getCheckpointPolicy, getPhaseOneModels
from .phase_one import renderPlanAsJson, renderPlanAsMarkdown


DEFAULT_DATASET_PATH = Path("gt-harmbench-dataset/gt-harmbench-with-targets.csv")


def addDatasetSummaryCommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "dataset-summary",
        help="Print a compact summary of the local GT-HarmBench dataset.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the GT-HarmBench CSV file.",
    )


def addModelRegistryCommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    subparsers.add_parser(
        "model-registry",
        help="Print the Qwen-only phase-one model registry and checkpoint policy.",
    )


def addPhaseOnePlanCommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "phase-one-plan",
        help="Render the workshop-oriented phase-one analysis plan.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )


def formatDatasetSummary(csv_path: Path) -> str:
    records = loadPromptRecords(csv_path)
    summary = summarizeDataset(records)
    lines: list[str] = []
    lines.append(f"rows: {summary.row_count}")
    lines.append(f"csv: {csv_path}")
    lines.append("")
    lines.append("games:")
    for game, count in summary.game_counts.items():
        lines.append(f"  - {game}: {count}")
    lines.append("")
    lines.append("risk levels:")
    for risk_level, count in summary.risk_counts.items():
        lines.append(f"  - {risk_level}: {count}")
    lines.append("")
    lines.append("target disagreement buckets:")
    for bucket, count in summary.disagreement_counts.items():
        lines.append(f"  - {bucket}: {count}")
    return "\n".join(lines)


def formatModelRegistry() -> str:
    checkpoint_policy = getCheckpointPolicy()
    lines: list[str] = []
    lines.append("phase-one models:")
    for model in getPhaseOneModels():
        lines.append(
            f"  - {model.key}: {model.repo_id} | objective={model.objective} | "
            f"inference={model.inference_condition} | format={model.adapter_format}"
        )
    lines.append("")
    lines.append(f"checkpoint policy: {checkpoint_policy.name}")
    lines.append(f"checkpoint count: {checkpoint_policy.checkpoint_count}")
    lines.append("target events:")
    for event in checkpoint_policy.target_events:
        lines.append(f"  - {event}")
    lines.append("")
    lines.append(f"selection rule: {checkpoint_policy.selection_rule}")
    return "\n".join(lines)


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ama-mech-interp",
        description="Mechanistic interpretability helpers for agentic moral alignment.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    addDatasetSummaryCommand(subparsers)
    addModelRegistryCommand(subparsers)
    addPhaseOnePlanCommand(subparsers)
    return parser


def main() -> None:
    parser = buildParser()
    args = parser.parse_args()

    if args.command == "dataset-summary":
        print(formatDatasetSummary(args.csv))
        return

    if args.command == "model-registry":
        print(formatModelRegistry())
        return

    if args.command == "phase-one-plan":
        if args.format == "json":
            print(renderPlanAsJson())
        else:
            print(renderPlanAsMarkdown())
        return

    parser.error(f"Unknown command: {args.command}")
