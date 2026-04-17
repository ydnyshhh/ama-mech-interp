from __future__ import annotations

from pathlib import Path

__path__ = [str(Path(__file__).with_suffix(""))]

from .models.load_qwen_adapter import getPhaseOneModelLoadSpecs


def getPhaseOneModels() -> list[object]:
    return getPhaseOneModelLoadSpecs()


__all__ = ["getPhaseOneModels"]
