"""Conservative, review-required reuse proposals from Stage 4.1 evidence."""
import ast
import re


def _source(result, filename):
    rag = ((result.get('stage41_evidence') or {}).get('retrieval_evidence') or {}).get('rag_context') or []
    for item in rag:
        if item.get('sd_file') != filename:
            continue
        for source in item.get('source_code') or []:
            if isinstance(source, str):
                try:
                    ast.parse(source)
                except SyntaxError:
                    continue
                return source
    return ''


def propose_reuse(result, functions, signals):
    """No inference from similar SDs; only exact existing-pattern extension."""
    ext = ((result.get('stage41_evidence') or {}).get('existing_pattern_extension') or {})
    if not ext or ext.get('requested_step_type') != 'then':
        return None
    source = _source(result, ext.get('sd_file'))
    mappings = {}
    if source:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                    continue
                if not isinstance(value, dict) or not value or not all(isinstance(k, str) and isinstance(v, (str, int, float, bool)) for k,v in value.items()):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ('status_dict', 'state_dict', 'value_map', 'status_map'):
                        mappings = {k: str(v) for k,v in value.items()}
                        break
    pattern = ext.get('pattern') or ''
    placeholders = re.findall(r'\{([A-Za-z_]\w*)\}', pattern)
    if len(placeholders) != 1:
        return None
    signal_candidates = [key for key in ext.get('observed_signals', []) if key in signals]
    signal = signal_candidates[0] if len(signal_candidates) == 1 else None
    api = next((key for key in ('check_signal','compare_signal') if key in functions), None)
    return {'pattern': pattern, 'placeholder': placeholders[0], 'mapping': mappings,
            'signal': signal, 'api': api, 'source_file': ext.get('sd_file'),
            'signal_candidates': ext.get('observed_signals', []),
            'requires_confirmation': True}


def apply_reuse_to_steps(steps, proposal, functions):
    """Populate only catalogue-backed fields, never invent a signal key."""
    if not proposal:
        return steps
    for step in steps:
        if step.get('api_key') != proposal['api']:
            continue
        params = step.get('parameters') or {}
        if 'String.Variable' in params and proposal['signal']:
            params['String.Variable'] = proposal['signal']
        if 'Variant.ExpectedValue' in params:
            params['Variant.ExpectedValue'] = '{' + proposal['placeholder'] + '}'
    return steps
