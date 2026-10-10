from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List


def parse_feature_file(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    feature = {"feature_file": path.name, "source_path": str(path), "feature": None, "scenarios": []}
    current = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Feature:"):
            feature["feature"] = stripped.split(":", 1)[1].strip()
        elif re.match(r"Scenario(?: Outline)?:", stripped):
            current = {"name": stripped.split(":", 1)[1].strip(), "steps": []}
            feature["scenarios"].append(current)
        elif re.match(r"(?:Given|When|Then|And|But)\s+", stripped) and current is not None:
            m = re.match(r"(Given|When|Then|And|But)\s+(.*)", stripped)
            current["steps"].append({"keyword": m.group(1), "text": m.group(2)})
    return feature
