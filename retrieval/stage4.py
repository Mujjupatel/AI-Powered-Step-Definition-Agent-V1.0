import json, re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .retriever import HybridRetriever

DEFAULT_TYPE_WEIGHTS = {
    'keyword_pattern': 1.28,
    'source_code': 1.14,
    'api_usage': 1.12,
    'signal_usage': 1.10,
    'parameter_mapping': 1.08,
    'execution_flow': 1.04,
    'value_mapping': 1.03,
    'gherkin_scenario': 0.72,
}

class Stage4Retriever:
    """Type-aware hybrid retrieval plus authoritative catalogue enrichment.

    Semantic retrieval discovers similar implementation patterns. Function and signal
    catalogue records are never synthesized: only exact keys present in the Stage 2
    catalogue snapshot are returned as authoritative records.
    """
    def __init__(self, index_dir='rag_index', catalogue_path='example_output/catalogue_snapshot.json'):
        self.base = HybridRetriever(index_dir)
        self.catalogue_path = Path(catalogue_path)
        self.catalogue = json.loads(self.catalogue_path.read_text(encoding='utf-8'))
        self.functions = self.catalogue.get('functions', {})
        self.signals = self.catalogue.get('signals', {})
        self.rows_by_file = defaultdict(list)
        for row in self.base.rows:
            self.rows_by_file[row['source_file']].append(row)

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()

    def _exact_mentions(self, query: str, catalogue: Dict[str, Any]) -> List[str]:
        q = f" {self._norm(query)} "
        found=[]
        for key, record in catalogue.items():
            candidates=[key, key.replace('_',' ')]
            for field in ('api_name','signal_name','name'):
                if record.get(field): candidates.append(str(record[field]))
            if any(f" {self._norm(c)} " in q for c in candidates if c):
                found.append(key)
        return found

    def _weighted_candidates(self, query: str, candidate_k: int = 40) -> List[Dict[str, Any]]:
        raw=self.base.search(query, top_k=candidate_k, semantic_weight=0.72)
        out=[]
        for r in raw:
            x=dict(r)
            weight=DEFAULT_TYPE_WEIGHTS.get(x['chunk_type'],1.0)
            x['type_weight']=weight
            x['stage4_score']=round(max(0.0, x['score']) * weight, 6)
            out.append(x)
        return sorted(out, key=lambda x:x['stage4_score'], reverse=True)

    def _rank_files(self, candidates: List[Dict[str, Any]], top_files: int) -> List[Dict[str, Any]]:
        groups=defaultdict(list)
        for r in candidates:
            if r['source_file'].endswith('.py'):
                groups[r['source_file']].append(r)
        ranked=[]
        for source, rows in groups.items():
            rows=sorted(rows,key=lambda x:x['stage4_score'],reverse=True)
            # Best evidence dominates; diverse supporting chunk types add a bounded bonus.
            best=rows[0]['stage4_score']
            distinct={r['chunk_type']:r for r in rows}
            support=sum(sorted((r['stage4_score'] for r in distinct.values()),reverse=True)[1:4])
            file_score=best + 0.12*support
            ranked.append({'source_file':source,'score':round(file_score,6),'best_chunk':rows[0],
                           'matched_chunk_types':list(distinct.keys())})
        return sorted(ranked,key=lambda x:x['score'],reverse=True)[:top_files]

    def _file_evidence(self, source_file: str) -> Dict[str, List[Dict[str, Any]]]:
        evidence=defaultdict(list)
        for r in self.rows_by_file.get(source_file,[]):
            evidence[r['chunk_type']].append(r)
        return dict(evidence)

    def _catalogue_keys_from_evidence(self, evidence_by_file: Dict[str, Dict[str,List[Dict[str,Any]]]]):
        apis=[]; signals=[]
        for evidence in evidence_by_file.values():
            for r in evidence.get('api_usage',[]):
                k=r.get('metadata',{}).get('api_key')
                if k and k not in apis: apis.append(k)
            for r in evidence.get('signal_usage',[]):
                k=r.get('metadata',{}).get('signal_key')
                if k and k not in signals: signals.append(k)
        return apis,signals

    @staticmethod
    def _confidence(score: float) -> float:
        # Heuristic ranking confidence, intentionally not presented as calibrated probability.
        return round(min(0.99,max(0.0,score)),4)

    def query(self, query: str, top_files: int=3, candidate_k: int=40) -> Dict[str,Any]:
        candidates=self._weighted_candidates(query,candidate_k)
        files=self._rank_files(candidates,top_files)
        evidence={f['source_file']:self._file_evidence(f['source_file']) for f in files}
        # Only the best-matching SD is allowed to seed generation catalogue records.
        # Lower-ranked SDs remain supporting pattern evidence and cannot leak unrelated APIs/signals.
        primary_evidence = {files[0]['source_file']: evidence[files[0]['source_file']]} if files else {}
        ev_api,ev_signal=self._catalogue_keys_from_evidence(primary_evidence)
        exact_api=self._exact_mentions(query,self.functions)
        exact_signal=self._exact_mentions(query,self.signals)
        api_keys=list(dict.fromkeys(exact_api+ev_api))
        signal_keys=list(dict.fromkeys(exact_signal+ev_signal))

        best_keyword=None
        for f in files:
            patterns=evidence[f['source_file']].get('keyword_pattern',[])
            if patterns:
                # Re-score keyword chunks directly so the selected pattern is query-relevant.
                scored=self.base.search(query,top_k=50,chunk_types=['keyword_pattern'])
                chosen=next((r for r in scored if r['source_file']==f['source_file']),patterns[0])
                best_keyword={
                    'sd_file':f['source_file'],
                    'pattern':chosen.get('metadata',{}).get('keyword_pattern'),
                    'step_type':chosen.get('metadata',{}).get('step_type'),
                    'confidence':self._confidence(f['score']),
                    'confidence_note':'heuristic retrieval score; not a calibrated probability',
                }
                break

        similar=[]
        for f in files:
            similar.append({'sd_file':f['source_file'],'score':f['score'],
                            'matched_chunk_types':f['matched_chunk_types']})

        context_files=[]
        for f in files:
            ev=evidence[f['source_file']]
            context_files.append({
                'sd_file':f['source_file'],
                'rank_score':f['score'],
                'keyword_patterns':[r['text'] for r in ev.get('keyword_pattern',[])],
                'api_usage':[r['text'] for r in ev.get('api_usage',[])],
                'signal_usage':[r['text'] for r in ev.get('signal_usage',[])],
                'parameter_mapping':[r['text'] for r in ev.get('parameter_mapping',[])],
                'execution_flow':[r['text'] for r in ev.get('execution_flow',[])],
                'value_mapping':[r['text'] for r in ev.get('value_mapping',[])],
                'source_code':[r['text'] for r in ev.get('source_code',[])],
            })

        return {
            'schema_version':'stage4-v1',
            'query':query,
            'best_matching_keyword':best_keyword,
            'similar_sd_files':similar,
            'authoritative_catalogue':{
                'functions':[{**self.functions[k],'source':'Function Catalogue'} for k in api_keys if k in self.functions],
                'signals':[{'signal_key':k, **self.signals[k], 'source':'Signal Catalogue'} for k in signal_keys if k in self.signals],
                'exact_query_function_keys':exact_api,
                'exact_query_signal_keys':exact_signal,
                'policy':'Only records present in catalogue_snapshot.json are authoritative; missing keys must not be invented.'
            },
            'rag_context':context_files,
            'top_weighted_chunks':[
                {k:r[k] for k in ('source_file','chunk_type','stage4_score','score','semantic_score','lexical_score','text')}
                for r in candidates[:10]
            ],
            'generation_guardrails':[
                'Use catalogue API keys only from authoritative_catalogue.functions.',
                'Use signal keys/model paths only from authoritative_catalogue.signals.',
                'Treat retrieved SD files as implementation-pattern evidence, not catalogue authority.',
                'If required API/signal information is absent, ask the user instead of inventing it.',
                'Preserve observed source inconsistencies as observations; do not silently correct source knowledge.'
            ]
        }
