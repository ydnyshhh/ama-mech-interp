from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any


def ensureDirectory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensureParentDirectory(path: Path) -> Path:
    ensureDirectory(path.parent)
    return path


def convertToJsonReady(value: Any) -> Any:
    if is_dataclass(value):
        return convertToJsonReady(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [convertToJsonReady(item) for item in value]
    if isinstance(value, list):
        return [convertToJsonReady(item) for item in value]
    if isinstance(value, dict):
        return {str(key): convertToJsonReady(item) for key, item in value.items()}
    return value


def writeJsonFile(path: Path, payload: Any) -> Path:
    ensureParentDirectory(path)
    path.write_text(json.dumps(convertToJsonReady(payload), indent=2), encoding="utf-8")
    return path


def writeJsonlFile(path: Path, rows: list[Any]) -> Path:
    ensureParentDirectory(path)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(convertToJsonReady(row)))
            handle.write("\n")
    return path


def readJsonFile(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def readJsonlFile(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            cleaned_line = line.strip()
            if not cleaned_line:
                continue
            rows.append(json.loads(cleaned_line))
    return rows
