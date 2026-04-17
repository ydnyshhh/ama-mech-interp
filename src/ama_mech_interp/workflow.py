from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .analysis.behavior_summary import writeBehaviorSummaryJobSpec
from .analysis.disagreement import writeDisagreementJobSpec
from .analysis.drift import writeDriftJobSpec
from .analysis.logit_depth import writeLogitDepthJobSpec
from .analysis.probes import writeProbeJobSpec
from .data.prompt_suite import buildDefaultPromptSuite, buildMinimumViablePromptSuite, writePromptSuite
from .eval.run_behavior import writeBehaviorBundle
from .extract.save_activations import writeActivationPlan
from .models.load_qwen_adapter import buildRunSpec, getMinimumViableModelLoadSpecs
from .storage import readJsonlFile, writeJsonFile


@dataclass(frozen=True)
class MaterializedPromptSuites:
    full_prompt_suite_path: Path
    minimum_viable_prompt_suite_path: Path
    full_prompt_count: int
    minimum_viable_prompt_count: int


def materializePromptSuites(csv_path: Path, artifacts_root: Path) -> MaterializedPromptSuites:
    full_prompt_suite_rows = buildDefaultPromptSuite(csv_path)
    minimum_viable_rows = buildMinimumViablePromptSuite(full_prompt_suite_rows)
    full_prompt_suite_path = artifacts_root / "prompt_suite_v1.jsonl"
    minimum_viable_prompt_suite_path = artifacts_root / "prompt_suite_mvr1.jsonl"
    writePromptSuite(full_prompt_suite_path, full_prompt_suite_rows)
    writePromptSuite(minimum_viable_prompt_suite_path, minimum_viable_rows)
    writeJsonFile(
        artifacts_root / "prompt_suite_manifest.json",
        {
            "full_prompt_suite_path": str(full_prompt_suite_path),
            "minimum_viable_prompt_suite_path": str(minimum_viable_prompt_suite_path),
            "full_prompt_count": len(full_prompt_suite_rows),
            "minimum_viable_prompt_count": len(minimum_viable_rows),
        },
    )
    return MaterializedPromptSuites(
        full_prompt_suite_path=full_prompt_suite_path,
        minimum_viable_prompt_suite_path=minimum_viable_prompt_suite_path,
        full_prompt_count=len(full_prompt_suite_rows),
        minimum_viable_prompt_count=len(minimum_viable_rows),
    )


def buildAnalysisGroupKey(prompt_suite_name: str, run_id: str) -> str:
    return f"{prompt_suite_name}__{run_id}"


def bootstrapMinimumViablePipeline(
    csv_path: Path,
    artifacts_root: Path,
    outputs_root: Path,
    run_id: str = "mvr1",
) -> dict[str, object]:
    materialized_prompt_suites = materializePromptSuites(csv_path, artifacts_root)
    minimum_viable_prompt_rows = readJsonlFile(materialized_prompt_suites.minimum_viable_prompt_suite_path)
    prompt_suite_name = "prompt_suite_mvr1"

    behavior_outputs: list[dict[str, str]] = []
    activation_outputs: list[dict[str, str]] = []
    activation_plan_paths: list[Path] = []
    behavior_rows_paths: list[Path] = []

    for model_spec in getMinimumViableModelLoadSpecs():
        run_spec = buildRunSpec(model_spec, prompt_suite=prompt_suite_name, run_id=run_id)
        behavior_output = writeBehaviorBundle(
            output_root=outputs_root,
            spec=run_spec,
            prompt_suite_path=materialized_prompt_suites.minimum_viable_prompt_suite_path,
            prompt_suite_rows=minimum_viable_prompt_rows,
        )
        behavior_outputs.append(behavior_output)
        behavior_rows_paths.append(Path(behavior_output["behavior_rows_path"]))
        activation_output = writeActivationPlan(
            output_root=outputs_root,
            spec=run_spec,
            behavior_rows=readJsonlFile(Path(behavior_output["behavior_rows_path"])),
        )
        activation_outputs.append(activation_output)
        activation_plan_paths.append(Path(activation_output["activation_plan_path"]))

    analysis_group_key = buildAnalysisGroupKey(prompt_suite_name, run_id)
    behavior_summary_job_path = outputs_root / "behavior" / analysis_group_key / "behavior_summary_job.json"
    drift_job_path = outputs_root / "drift" / analysis_group_key / "drift_job.json"
    probes_job_path = outputs_root / "probes" / analysis_group_key / "probes_job.json"
    logit_depth_job_path = outputs_root / "logit_depth" / analysis_group_key / "logit_depth_job.json"
    disagreement_job_path = outputs_root / "disagreement" / analysis_group_key / "disagreement_job.json"

    writeBehaviorSummaryJobSpec(
        output_path=behavior_summary_job_path,
        prompt_suite_path=materialized_prompt_suites.minimum_viable_prompt_suite_path,
        behavior_rows_paths=behavior_rows_paths,
    )

    writeDriftJobSpec(
        output_path=drift_job_path,
        prompt_suite_path=materialized_prompt_suites.minimum_viable_prompt_suite_path,
        activation_plan_paths=activation_plan_paths,
    )
    writeProbeJobSpec(
        output_path=probes_job_path,
        prompt_suite_path=materialized_prompt_suites.minimum_viable_prompt_suite_path,
        activation_plan_paths=activation_plan_paths,
    )
    writeLogitDepthJobSpec(
        output_path=logit_depth_job_path,
        prompt_suite_path=materialized_prompt_suites.minimum_viable_prompt_suite_path,
        activation_plan_paths=activation_plan_paths,
    )
    writeDisagreementJobSpec(
        output_path=disagreement_job_path,
        prompt_suite_path=materialized_prompt_suites.minimum_viable_prompt_suite_path,
        behavior_rows_paths=behavior_rows_paths,
    )

    manifest = {
        "run_id": run_id,
        "prompt_suite_name": prompt_suite_name,
        "materialized_prompt_suites": materialized_prompt_suites,
        "behavior_outputs": behavior_outputs,
        "activation_outputs": activation_outputs,
        "analysis_jobs": {
            "behavior_summary": str(behavior_summary_job_path),
            "drift": str(drift_job_path),
            "probes": str(probes_job_path),
            "logit_depth": str(logit_depth_job_path),
            "disagreement": str(disagreement_job_path),
        },
    }
    writeJsonFile(outputs_root / "minimum_viable_pipeline.json", manifest)
    return manifest
