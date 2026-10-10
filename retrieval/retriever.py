import json, re
from pathlib import Path
import numpy as np
from .embeddings import create_backend

class HybridRetriever:
    def __init__(self,index_dir):
        self.index_dir=Path(index_dir)
        self.matrix=np.load(self.index_dir/'embeddings.npy')
        self.rows=[json.loads(x) for x in (self.index_dir/'metadata.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
        manifest=json.loads((self.index_dir/'index_manifest.json').read_text(encoding='utf-8'))
        self.provider=manifest['embedding_provider']
        model_name='sentence-transformers/all-MiniLM-L6-v2'
        if self.provider=='sentence-transformers':
            p=self.index_dir/'embedding_model'/'model_name.txt'
            model_name=p.read_text(encoding='utf-8').strip()
        self.backend=create_backend(self.provider,model_name)
        if self.provider=='tfidf': self.backend=type(self.backend).load(self.index_dir/'embedding_model')

    @staticmethod
    def _tokens(text): return set(re.findall(r'[a-zA-Z0-9_]+', text.lower()))
    def _lexical_score(self,q,text):
        qt=self._tokens(q); tt=self._tokens(text)
        if not qt or not tt: return 0.0
        return len(qt & tt)/len(qt)
    def search(self, query, top_k=5, chunk_types=None, metadata_filters=None, semantic_weight=0.75):
        qv=self.backend.encode([query])[0]
        qn=np.linalg.norm(qv); qv=qv/max(qn,1e-12)
        semantic=self.matrix @ qv
        scores=[]
        for i,row in enumerate(self.rows):
            if chunk_types and row['chunk_type'] not in chunk_types: continue
            md=row.get('metadata',{})
            if metadata_filters and any(str(md.get(k,'')) != str(v) for k,v in metadata_filters.items()): continue
            lexical=self._lexical_score(query,row['text'])
            score=semantic_weight*float(semantic[i])+(1-semantic_weight)*lexical
            scores.append((score,i,float(semantic[i]),lexical))
        scores.sort(reverse=True)
        out=[]
        for score,i,sem,lex in scores[:top_k]:
            row=dict(self.rows[i]); row['score']=round(score,6); row['semantic_score']=round(sem,6); row['lexical_score']=round(lex,6); out.append(row)
        return out
