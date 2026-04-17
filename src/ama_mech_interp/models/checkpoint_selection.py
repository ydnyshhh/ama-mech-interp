from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from ..storage import writeJsonFile


@dataclass(frozen=True)
class CheckpointCandidate:
    checkpoint_label: str
    step: int
    behavior_score: float | None


@dataclass(frozen=True)
class CheckpointSelection:
    selection_name: str
    selected_checkpoints: tuple[str, ...]
    selection_reason: str


def loadCheckpointCandidates(path: Path) -> list[CheckpointCandidate]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates: list[CheckpointCandidate] = []
    for row in payload:
        candidates.append(
            CheckpointCandidate(
                checkpoint_label=str(row["checkpoint_label"]),
                step=int(row["step"]),
                behavior_score=float(row["behavior_score"]) if row.get("behavior_score") is not None else None,
            )
        )
    candidates.sort(key=lambda candidate: candidate.step)
    return candidates


def selectSparseCheckpointLadder(candidates: list[CheckpointCandidate], checkpoint_count: int = 5) -> CheckpointSelection:
    if len(candidates) <= checkpoint_count:
        return CheckpointSelection(
            selection_name="all_candidates",
            selected_checkpoints=tuple(candidate.checkpoint_label for candidate in candidates),
            selection_reason="Candidate count was already within the requested ladder size.",
        )

    selected_indices: list[int] = [0]
    final_index = len(candidates) - 1

    behavior_scored_candidates = [candidate for candidate in candidates if candidate.behavior_score is not None]
    behavior_priority_indices: list[int] = []
    if len(behavior_scored_candidates) >= 3:
        deltas: list[tuple[float, int]] = []
        for index in range(1, len(candidates)):
            previous_candidate = candidates[index - 1]
            current_candidate = candidates[index]
            if previous_candidate.behavior_score is None or current_candidate.behavior_score is None:
                continue
            delta = abs(current_candidate.behavior_score - previous_candidate.behavior_score)
            deltas.append((delta, index))
        deltas.sort(reverse=True)
        for _, index in deltas[:2]:
            behavior_priority_indices.extend([max(index - 1, 0), index])

    evenly_spaced_indices = [
        round(step_index * final_index / (checkpoint_count - 1))
        for step_index in range(checkpoint_count)
    ]

    for index in [*behavior_priority_indices, *evenly_spaced_indices]:
        if index in selected_indices or index == final_index:
            continue
        if len(selected_indices) >= checkpoint_count - 1:
            break
        selected_indices.append(index)

    if final_index not in selected_indices:
        selected_indices.append(final_index)

    ordered_indices = sorted(selected_indices)[:checkpoint_count]
    selected_checkpoints = tuple(candidates[index].checkpoint_label for index in ordered_indices)
    return CheckpointSelection(
        selection_name="behavior_anchored_sparse_ladder",
        selected_checkpoints=selected_checkpoints,
        selection_reason=(
            "Selected endpoints first, then behavior-shift candidates when scores were available, "
            "and filled remaining slots with evenly spread checkpoints."
        ),
    )


def writeCheckpointSelection(path: Path, selection: CheckpointSelection) -> Path:
    return writeJsonFile(path, selection)
