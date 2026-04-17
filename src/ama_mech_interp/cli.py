from __future__ import annotations

import argparse
from collections import Counter
import logging
from pathlib import Path

from .analysis.behavior_summary import writeBehaviorSummary
from .analysis.disagreement import writeDisagreementArtifacts
from .data.gt_harmbench import loadGtHarmBenchDataset
from .eval.run_behavior import executeBehaviorRun, executeBehaviorRunFromManifest
from .models.checkpoint_selection import loadCheckpointCandidates, selectSparseCheckpointLadder, writeCheckpointSelection
from .models.load_qwen_adapter import InferenceLoadConfig, buildRunSpec, getModelLoadSpec, getPhaseOneModelLoadSpecs
from .phase_one import renderPlanAsJson, renderPlanAsMarkdown
from .storage import readJsonlFile
from .workflow import bootstrapMinimumViablePipeline, materializePromptSuites


DEFAULT_DATASET_PATH = Path("gt-harmbench-dataset/gt-harmbench-with-targets.csv")
DEFAULT_ARTIFACTS_ROOT = Path("artifacts")
DEFAULT_OUTPUTS_ROOT = Path("outputs")


def addDatasetSummaryCommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("dataset-summary", help="Print a compact summary of the local GT-HarmBench dataset.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATASET_PATH, help="Path to the GT-HarmBench CSV file.")


def addModelRegistryCommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    subparsers.add_parser("model-registry", help="Print the Qwen phase-one model loader registry.")


def addPhaseOnePlanCommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("phase-one-plan", help="Render the workshop-oriented phase-one analysis plan.")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown", help="Output format.")


def addMaterializePromptSuiteCommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("materialize-prompt-suite", help="Build and write the canonical prompt suite artifacts.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATASET_PATH, help="Path to the GT-HarmBench CSV file.")
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT, help="Directory for prompt suite artifacts.")
    parser.add_argument("--skip-csv", action="store_true", help="Do not write companion CSV artifacts.")


def addMinimumViablePipelineCommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("minimum-viable-pipeline", help="Materialize the prompt suite and prepare the minimum viable first-result pipeline.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATASET_PATH, help="Path to the GT-HarmBench CSV file.")
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT, help="Directory for prompt suite artifacts.")
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT, help="Directory for pipeline outputs.")
    parser.add_argument("--run-id", default="mvr1", help="Run id for the minimum viable bundle.")


def addCheckpointSelectionCommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("select-checkpoints", help="Select a sparse checkpoint ladder from a JSON or JSONL candidate list.")
    parser.add_argument("--input", type=Path, required=True, help="Path to checkpoint candidates in JSON or JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Path for the selected checkpoint JSON.")


def addBehaviorSummaryCommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("behavior-summary", help="Write a behavior summary artifact from a behavior rows file.")
    parser.add_argument("--prompt-suite", type=Path, required=True, help="Path to the prompt suite JSONL.")
    parser.add_argument("--behavior-rows", type=Path, required=True, help="Path to behavior rows JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Path for the summary JSON.")


def addRunBehaviorCommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("run-behavior", help="Execute actual behavior inference for a planned manifest or a model-key plus prompt-suite pair.")
    parser.add_argument("--manifest", type=Path, help="Path to an existing behavior run manifest.")
    parser.add_argument("--model-key", help="Phase-one model key such as qwen_base or qwen_deont_tool.")
    parser.add_argument("--prompt-suite", type=Path, help="Path to the prompt suite JSONL.")
    parser.add_argument("--run-id", default="manual", help="Run id used in the canonical run key.")
    parser.add_argument("--outputs-root", type=Path, help="Directory for behavior outputs. Defaults to the manifest root when --manifest is used, otherwise outputs.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Torch dtype for model loading.",
    )
    parser.add_argument("--local-files-only", action="store_true", help="Load model files only from the local Hugging Face cache.")
    parser.add_argument("--force", action="store_true", help="Recompute completed prompts instead of resuming.")
    parser.add_argument("--max-prompts", type=int, help="Execute only the first N pending prompts.")


def addDisagreementCommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("disagreement", help="Rank disagreement prompts and write the pairwise disagreement matrix.")
    parser.add_argument("--prompt-suite", type=Path, required=True, help="Path to the prompt suite JSONL.")
    parser.add_argument("--behavior-rows", type=Path, nargs="+", required=True, help="One or more behavior rows JSONL files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for disagreement artifacts.")


def formatDatasetSummary(csv_path: Path) -> str:
    load_result = loadGtHarmBenchDataset(csv_path)
    rows = load_result.records
    report = load_result.report
    game_counts = Counter(row.formal_game for row in rows)
    risk_counts = Counter(str(row.risk_level) if row.risk_level is not None else "missing" for row in rows)
    disagreement_counts = Counter(row.target_agreement_bucket for row in rows)
    lines: list[str] = [
        f"rows: {len(rows)}",
        f"csv: {csv_path}",
        f"dropped_rows: {report.dropped_rows}",
        "",
        "games:",
    ]
    for game, count in game_counts.most_common():
        lines.append(f"  - {game}: {count}")
    lines.append("")
    lines.append("risk levels:")
    for risk_level, count in risk_counts.most_common():
        lines.append(f"  - {risk_level}: {count}")
    lines.append("")
    lines.append("target disagreement buckets:")
    for bucket, count in disagreement_counts.most_common():
        lines.append(f"  - {bucket}: {count}")
    return "\n".join(lines)


def formatModelRegistry() -> str:
    lines: list[str] = ["phase-one models:"]
    for model in getPhaseOneModelLoadSpecs():
        lines.append(
            f"  - {model.model_key}: {model.repo_id} | objective={model.objective} | "
            f"reasoning={model.reasoning_mode} | tool={model.tool_mode} | checkpoint={model.checkpoint}"
        )
    return "\n".join(lines)


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ama-mech-interp", description="Mechanistic interpretability helpers for agentic moral alignment.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    addDatasetSummaryCommand(subparsers)
    addModelRegistryCommand(subparsers)
    addPhaseOnePlanCommand(subparsers)
    addMaterializePromptSuiteCommand(subparsers)
    addMinimumViablePipelineCommand(subparsers)
    addCheckpointSelectionCommand(subparsers)
    addBehaviorSummaryCommand(subparsers)
    addRunBehaviorCommand(subparsers)
    addDisagreementCommand(subparsers)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = buildParser()
    args = parser.parse_args()

    if args.command == "dataset-summary":
        print(formatDatasetSummary(args.csv))
        return

    if args.command == "model-registry":
        print(formatModelRegistry())
        return

    if args.command == "phase-one-plan":
        print(renderPlanAsJson() if args.format == "json" else renderPlanAsMarkdown())
        return

    if args.command == "materialize-prompt-suite":
        result = materializePromptSuites(args.csv, args.artifacts_root, write_companion_csv=not args.skip_csv)
        print(f"master_prompt_pool: {result.master_prompt_pool_path}")
        print(f"full_prompt_suite: {result.full_prompt_suite_path}")
        print(f"minimum_viable_prompt_suite: {result.minimum_viable_prompt_suite_path}")
        print(f"master_prompt_count: {result.master_prompt_count}")
        print(f"full_prompt_count: {result.full_prompt_count}")
        print(f"minimum_viable_prompt_count: {result.minimum_viable_prompt_count}")
        print(f"build_report: {result.build_report_path}")
        return

    if args.command == "minimum-viable-pipeline":
        manifest = bootstrapMinimumViablePipeline(
            csv_path=args.csv,
            artifacts_root=args.artifacts_root,
            outputs_root=args.outputs_root,
            run_id=args.run_id,
        )
        print(f"pipeline_manifest: {args.outputs_root / 'minimum_viable_pipeline.json'}")
        print(f"minimum_viable_prompt_count: {manifest['materialized_prompt_suites'].minimum_viable_prompt_count}")
        print(f"behavior_runs: {len(manifest['behavior_outputs'])}")
        for command in manifest["behavior_execution_commands"]:
            print(f"run_behavior_command: {command}")
        return

    if args.command == "select-checkpoints":
        candidates = loadCheckpointCandidates(args.input)
        selection = selectSparseCheckpointLadder(candidates)
        writeCheckpointSelection(args.output, selection)
        print(f"selected_checkpoints: {', '.join(selection.selected_checkpoints)}")
        return

    if args.command == "behavior-summary":
        writeBehaviorSummary(args.output, args.prompt_suite, args.behavior_rows)
        print(f"behavior_summary: {args.output}")
        return

    if args.command == "run-behavior":
        inference_config = InferenceLoadConfig(
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            local_files_only=args.local_files_only,
        )
        try:
            if args.manifest is not None:
                summary = executeBehaviorRunFromManifest(
                    manifest_path=args.manifest,
                    output_root=args.outputs_root,
                    inference_config=inference_config,
                    force=args.force,
                    max_prompts=args.max_prompts,
                    logger=logging.getLogger("ama_mech_interp.run_behavior"),
                )
            else:
                if not args.model_key or args.prompt_suite is None:
                    parser.error("run-behavior requires either --manifest or both --model-key and --prompt-suite.")
                run_spec = buildRunSpec(
                    model_spec=getModelLoadSpec(args.model_key),
                    prompt_suite=args.prompt_suite.stem,
                    run_id=args.run_id,
                )
                summary = executeBehaviorRun(
                    spec=run_spec,
                    prompt_suite_path=args.prompt_suite,
                    output_root=args.outputs_root or DEFAULT_OUTPUTS_ROOT,
                    inference_config=inference_config,
                    force=args.force,
                    max_prompts=args.max_prompts,
                    logger=logging.getLogger("ama_mech_interp.run_behavior"),
                )
        except RuntimeError as error:
            parser.exit(status=1, message=f"{error}\n")
        print(f"behavior_rows: {summary.behavior_rows_path}")
        print(f"run_manifest: {summary.manifest_path}")
        print(f"prompt_count: {summary.prompt_count}")
        print(f"executed_prompt_count: {summary.executed_prompt_count}")
        print(f"completed_prompt_count: {summary.completed_prompt_count}")
        print(f"failed_prompt_count: {summary.failed_prompt_count}")
        print(f"skipped_completed_count: {summary.skipped_completed_count}")
        print(f"remaining_planned_count: {summary.remaining_planned_count}")
        return

    if args.command == "disagreement":
        writeDisagreementArtifacts(args.output_dir, args.prompt_suite, args.behavior_rows)
        print(f"disagreement_output_dir: {args.output_dir}")
        return

    parser.error(f"Unknown command: {args.command}")
