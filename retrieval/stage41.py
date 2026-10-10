import re
from typing import Any, Dict, List
from .stage4 import Stage4Retriever


class Stage41Retriever(Stage4Retriever):
    """Stage 4.1: new-keyword classification and evidence safety.

    Exact existing SD implementations may confirm their observed APIs/signals.
    Similar SDs are evidence only. For new keywords, catalogue APIs are candidates
    selected by deterministic intent/type compatibility and never promoted to confirmed.
    """

    @staticmethod
    def _split_step_type(text: str):
        """Return (optional Gherkin decorator, keyword body)."""
        match = re.match(r"^\s*(given|when|then|and|but)\s+(.+?)\s*$", str(text), re.I)
        if match:
            return match.group(1).lower(), match.group(2)
        return None, str(text).strip()

    @staticmethod
    def _pattern_regex(pattern: str) -> re.Pattern:
        """Match a Behave pattern with placeholders against concrete keyword text.

        Quoted placeholders accept quoted values; unquoted placeholders accept
        one or more non-whitespace tokens. Static words and punctuation must match.
        """
        parts = []
        pos = 0
        placeholder = re.compile(r'([\"\']?)\{([A-Za-z_][A-Za-z0-9_]*)\}\1')
        for match in placeholder.finditer(pattern):
            parts.append(re.escape(pattern[pos:match.start()]))
            if match.group(1):
                quote = re.escape(match.group(1))
                parts.append(quote + r'[^\"\']+' + quote)
            else:
                parts.append(r'\S+(?:\s+\S+){0,3}')
            pos = match.end()
        parts.append(re.escape(pattern[pos:]))
        raw = ''.join(parts).replace(r'\ ', r'\s+')
        return re.compile(r'^\s*' + raw + r'\s*$', re.I)

    def _exact_existing_matches(self, query: str) -> List[Dict[str, Any]]:
        matches = []
        requested_type, body = self._split_step_type(query)
        for row in self.base.rows:
            if row.get('chunk_type') != 'keyword_pattern':
                continue
            metadata = row.get('metadata') or {}
            pattern = metadata.get('keyword_pattern')
            stored_type = str(metadata.get('step_type') or '').lower()
            if not pattern:
                continue
            _, stored_body = self._split_step_type(pattern)
            if requested_type in ('given', 'when', 'then') and stored_type != requested_type:
                continue
            if self._pattern_regex(stored_body).fullmatch(body):
                matches.append({'sd_file': row['source_file'], 'pattern': pattern,
                                'step_type': metadata.get('step_type'), 'row': row})
        return matches

    def _missing_decorator_match(self, query: str):
        """Find a matching SD pattern whose requested decorator is absent.

        Uses indexed keyword_pattern metadata, not semantic similarity.
        """
        requested, body = self._split_step_type(query)
        if requested not in ('given', 'when', 'then'):
            return None
        by_file = {}
        for row in self.base.rows:
            if row.get('chunk_type') != 'keyword_pattern':
                continue
            metadata = row.get('metadata') or {}
            pattern = metadata.get('keyword_pattern')
            step_type = str(metadata.get('step_type') or '').lower()
            if not pattern or step_type not in ('given', 'when', 'then'):
                continue
            _, stored_body = self._split_step_type(pattern)
            if not self._pattern_regex(stored_body).fullmatch(body):
                continue
            filename = row.get('source_file')
            if not filename:
                continue
            group = by_file.setdefault(filename, {'pattern': pattern, 'types': set()})
            group['types'].add(step_type)
        for filename, info in by_file.items():
            if requested in info['types']:
                continue
            ev = self._file_evidence(filename)
            api_keys, signal_keys = self._catalogue_keys_from_evidence({filename: ev})
            return {
                'sd_file': filename,
                'pattern': info['pattern'],
                'requested_step_type': requested,
                'available_step_types': sorted(info['types']),
                'observed_apis': list(dict.fromkeys(api_keys)),
                'observed_signals': list(dict.fromkeys(signal_keys)),
                'requires_separate_implementation_review': requested == 'then',
            }
        return None

    @staticmethod
    def _signal_type(record: Dict[str, Any]) -> str:
        return str(record.get('Type') or record.get('type') or '').lower()

    def _candidate_functions(self, query: str, known_signal_keys: List[str]) -> List[Dict[str, Any]]:
        q=self._norm(query)
        signal_types={self._signal_type(self.signals[k]) for k in known_signal_keys if k in self.signals}
        wants_set=bool(re.search(r'\b(set|sets|setting|change|changed|is set to|becomes?)\b', q))
        wants_check=bool(re.search(r'\b(should|verify|check|expected|equals?|reach)\b', q))
        wants_ramp=bool(re.search(r'\b(ramp|ramps|ramped|from.+to.+seconds?)\b', q))
        wants_toggle='toggle' in q
        scored=[]
        for key,rec in self.functions.items():
            score=0; reasons=[]
            inputs=set(rec.get('inputs') or [])
            purpose=self._norm(str(rec.get('purpose','')))
            if 'Variable' in inputs: score += 1
            if wants_ramp and key=='ramp_signal': score += 7; reasons.append('query expresses a ramp operation')
            if wants_check and key in ('check_signal','compare_signal'): score += 5; reasons.append('query expresses validation/check intent')
            if wants_toggle and key=='toggle_signal': score += 6; reasons.append('query expresses toggle intent')
            if wants_set and key=='set_signal': score += 6; reasons.append('query expresses generic signal-setting intent')
            if wants_set and key=='set_enum_signal' and 'enum' in signal_types: score += 7; reasons.append('known signal type is Enum')
            if key=='set_enum_signal' and signal_types and 'enum' not in signal_types: score -= 8
            if key=='toggle_signal' and signal_types and 'boolean' not in signal_types: score -= 5
            if key=='set_multiple_signals' and len(known_signal_keys)<=1: score -= 4
            if score >= 5:
                scored.append({'api_key':key, **rec, 'source':'Function Catalogue',
                               'status':'candidate_only', 'candidate_score':score,
                               'candidate_reason':'; '.join(reasons) or purpose})
        return sorted(scored,key=lambda x:x['candidate_score'],reverse=True)[:4]

    def query(self, query: str, top_files: int=3, candidate_k: int=40) -> Dict[str,Any]:
        base=super().query(query,top_files=top_files,candidate_k=candidate_k)
        exact_matches=self._exact_existing_matches(query)
        extension=None if exact_matches else self._missing_decorator_match(query)
        exact_signal=base['authoritative_catalogue']['exact_query_signal_keys']
        exact_api=base['authoritative_catalogue']['exact_query_function_keys']

        if exact_matches:
            status='EXACT_EXISTING'
            primary=exact_matches[0]['sd_file']
            ev=self._file_evidence(primary)
            confirmed_api,confirmed_signal=self._catalogue_keys_from_evidence({primary:ev})
            confirmed_api=list(dict.fromkeys(exact_api+confirmed_api))
            confirmed_signal=list(dict.fromkeys(exact_signal+confirmed_signal))
            candidates=[]
            requires_user=False
            missing=[]
        else:
            primary=None
            confirmed_api=list(exact_api)  # explicit API key/name in the user query only
            confirmed_signal=list(exact_signal)
            candidates=self._candidate_functions(query, confirmed_signal)
            if confirmed_signal:
                status='NEW_WITH_KNOWN_CATALOGUE'
            else:
                status='INSUFFICIENT_KNOWLEDGE'
            requires_user=True
            missing=[]
            if not confirmed_signal: missing.append('Required signal/catalogue key is unknown or not explicitly identifiable.')
            if not confirmed_api: missing.append('Implementation API is not confirmed for this new keyword.')
            missing += ['Confirm platform/TAMS decorator values.', 'Confirm operation-specific defaults such as PostDelay or Timeout when required.']

        if extension:
            status='EXISTING_PATTERN_MISSING_DECORATOR'
            primary=extension['sd_file']
            # Evidence from the existing implementation is contextual, not
            # authorization to reuse its setter API for a Then assertion.
            confirmed_api=list(exact_api)
            confirmed_signal=list(exact_signal)
            candidates=self._candidate_functions(query, confirmed_signal)
            if extension['requested_step_type']=='then':
                candidates=[c for c in candidates if c.get('api_key') not in
                            {'set_signal','set_enum_signal','toggle_signal',
                             'set_multiple_signals','ramp_signal'}]
            requires_user=True
            missing=['Review a suitable implementation for the requested decorator.',
                     'Confirm signal and verification API against the catalogues.',
                     'Engineer approval is required before updating an existing SD file.']

        # Reframe best match: never call a similar pattern the best matching keyword for a new keyword.
        if exact_matches:
            match=exact_matches[0]
            best={k:match[k] for k in ('sd_file','pattern','step_type')}
            best['confidence']=1.0
            best['confidence_note']='query structurally matches an existing parameterized keyword pattern'
        else:
            best=None

        known_signals=[{'signal_key':k, **self.signals[k], 'source':'Signal Catalogue', 'status':'confirmed_catalogue_match'}
                       for k in confirmed_signal if k in self.signals]
        confirmed_functions=[{**self.functions[k],'source':'Function Catalogue','status':'confirmed_implementation'}
                             for k in confirmed_api if k in self.functions]

        # Similar SDs are explicitly non-authoritative for implementation selection on new keywords.
        similar=[]
        for item in base['similar_sd_files']:
            x=dict(item)
            x['relationship']='exact_implementation' if exact_matches and item['sd_file']==primary else 'similar_pattern_evidence'
            x['implementation_authority']=bool(exact_matches and item['sd_file']==primary)
            similar.append(x)

        return {
            'schema_version':'stage4.1-v1',
            'query':query,
            'keyword_status':status,
            'exact_existing_keyword':bool(exact_matches),
            'best_matching_keyword':best,
            'existing_pattern_extension':extension,
            'known_signals':known_signals,
            'confirmed_apis':confirmed_functions,
            'candidate_apis':candidates,
            'requires_user_input':requires_user,
            'missing_information':missing,
            'similar_sd_files':similar,
            'retrieval_evidence':{
                'rag_context':base['rag_context'],
                'top_weighted_chunks':base['top_weighted_chunks'],
            },
            'catalogue_policy':{
                'api_existence':'Function Catalogue proves an API exists; it does not prove that API is correct for a new keyword.',
                'signal_authority':'Signal keys, TAMS IDs and model paths come only from Signal Catalogue.',
                'new_keyword_rule':'APIs inferred from similar SDs or intent/type compatibility remain candidate_only until confirmed.',
            },
            'generation_guardrails':[
                'EXACT_EXISTING may reuse the observed implementation pattern from its exact SD.',
                'NEW_WITH_KNOWN_CATALOGUE must not promote candidate APIs to confirmed APIs without user/approved planning evidence.',
                'INSUFFICIENT_KNOWLEDGE must request missing catalogue information instead of inventing it.',
                'Never invent signal keys, TAMS IDs, model paths, API keys, or API parameter schemas.',
            ]
        }
