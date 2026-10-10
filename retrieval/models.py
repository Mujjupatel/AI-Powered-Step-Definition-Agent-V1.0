from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class IndexedChunk:
    index_id: str
    chunk_id: str
    source_file: str
    chunk_type: str
    text: str
    metadata: Dict[str, Any]

    def to_dict(self):
        return asdict(self)
