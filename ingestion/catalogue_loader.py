from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_function_catalogue(path: str | Path) -> Dict[str, Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            raw = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported function catalogue format: {path}")

    if isinstance(raw, dict):
        raw = raw.get("apis", raw.get("functions", []))

    result: Dict[str, Dict[str, Any]] = {}
    for row in raw:
        key = row.get("api_key") or row.get("API Key") or row.get("key")
        if not key:
            continue
        if isinstance(row, dict):
            result[str(key)] = row
    return result


def load_signal_catalogue(path: str | Path) -> Dict[str, Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = row.get("Signal Key") or row.get("signal_key") or row.get("key")
        if not key:
            continue
        result[str(key)] = row
    return result


def catalogue_summary(functions: Dict[str, Dict[str, Any]], signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "function_api_count": len(functions),
        "signal_count": len(signals),
        "api_keys": sorted(functions),
        "signal_keys": sorted(signals),
    }
