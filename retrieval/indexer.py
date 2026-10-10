import hashlib, json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from .embeddings import create_backend
from .models import IndexedChunk


def _stable_id(chunk, ordinal):
    raw = f"{ordinal}|{chunk.get('chunk_id','')}|{chunk.get('source_file','')}|{chunk.get('chunk_type','')}|{chunk.get('text','')}"
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]

def load_chunks(path: Path):
    rows=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def build_index(chunks_path, output_dir, provider="tfidf", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    chunks_path=Path(chunks_path); output_dir=Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    raw=load_chunks(chunks_path)
    indexed=[]
    for i,c in enumerate(raw):
        indexed.append(IndexedChunk(_stable_id(c,i), c.get('chunk_id',''), c.get('source_file',''), c.get('chunk_type',''), c.get('text',''), c.get('metadata',{})))
    texts=[x.text for x in indexed]
    backend=create_backend(provider, model_name)
    backend.fit(texts)
    matrix=backend.encode(texts)
    if provider == 'tfidf':
        # normalize for cosine similarity
        norms=np.linalg.norm(matrix,axis=1,keepdims=True); matrix=matrix/np.maximum(norms,1e-12)
    np.save(output_dir/'embeddings.npy', matrix.astype(np.float32))
    with (output_dir/'metadata.jsonl').open('w',encoding='utf-8') as f:
        for x in indexed: f.write(json.dumps(x.to_dict(),ensure_ascii=False)+'\n')
    backend.save(output_dir/'embedding_model')
    manifest={
        'schema_version':'stage3-v1','source_chunks':str(chunks_path),'count':len(indexed),
        'embedding_provider':provider,'embedding_dimension':int(matrix.shape[1]) if len(matrix.shape)>1 else 0,
        'duplicate_source_chunk_ids':len(raw)-len({c.get('chunk_id') for c in raw}),
        'notes':['index_id is intentionally independent from Stage 2 chunk_id because duplicate chunk IDs may exist across step-type variants.']
    }
    (output_dir/'index_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest
