from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    file: Optional[str] = None
    line: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeChunk:
    chunk_id: str
    source_file: str
    chunk_type: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeCard:
    sd_file: str
    source_path: str
    source_code: str
    parse_status: str
    syntax_error: Optional[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    keyword: Dict[str, Any] = field(default_factory=dict)
    value_mappings: Dict[str, Any] = field(default_factory=dict)
    apis: List[Dict[str, Any]] = field(default_factory=list)
    signals: List[Dict[str, Any]] = field(default_factory=list)
    api_parameter_mapping: Dict[str, Any] = field(default_factory=dict)
    execution_sequence: List[Dict[str, Any]] = field(default_factory=list)
    context_outputs: List[str] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["issues"] = [i.to_dict() for i in self.issues]
        return data
