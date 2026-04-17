from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..run_spec import RunSpec, buildRunKey, getOutputDirectory
from ..storage import writeJsonFile, writeJsonlFile


@dataclass(frozen=True)
class ActivationRequest:
    run_key: str
    model_key: str
    checkpoint: str
    prompt_id: str
    tensor_name: str
    layer_selector: str
    decision_position: str


DEFAULT_TENSOR_NAMES = (
    "residual_stream_at_decision_position",
    "final_token_hidden_state_by_layer",
    "action_token_logits_and_logit_margins",
)


def buildActivationRequests(
    spec: RunSpec,
    behavior_rows: list[dict[str, object]],
    tensor_names: tuple[str, ...] = DEFAULT_TENSOR_NAMES,
    layer_selector: str = "all_layers",
) -> list[ActivationRequest]:
    run_key = buildRunKey(spec)
    requests: list[ActivationRequest] = []
    for behavior_row in behavior_rows:
        for tensor_name in tensor_names:
            requests.append(
                ActivationRequest(
                    run_key=run_key,
                    model_key=spec.model_key,
                    checkpoint=spec.checkpoint,
                    prompt_id=str(behavior_row["prompt_id"]),
                    tensor_name=tensor_name,
                    layer_selector=layer_selector,
                    decision_position="action_decision_position",
                )
            )
    return requests


def writeActivationPlan(
    output_root: Path,
    spec: RunSpec,
    behavior_rows: list[dict[str, object]],
    tensor_names: tuple[str, ...] = DEFAULT_TENSOR_NAMES,
) -> dict[str, str]:
    output_directory = getOutputDirectory(output_root, "activations", spec)
    activation_requests = buildActivationRequests(spec, behavior_rows, tensor_names=tensor_names)
    manifest = {
        "run_key": buildRunKey(spec),
        "run_spec": spec,
        "tensor_names": list(tensor_names),
        "activation_request_count": len(activation_requests),
        "layer_selector": "all_layers",
    }
    writeJsonFile(output_directory / "activation_plan.json", manifest)
    writeJsonlFile(output_directory / "activation_requests.jsonl", activation_requests)
    return {
        "output_directory": str(output_directory),
        "activation_plan_path": str(output_directory / "activation_plan.json"),
        "activation_requests_path": str(output_directory / "activation_requests.jsonl"),
    }
