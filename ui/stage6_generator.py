"""Deterministic Stage 6 Behave SD generator. No model-generated executable code."""
from __future__ import annotations
import ast
import json
import keyword
import re
from copy import deepcopy

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

class GenerationError(ValueError):
    pass


def _pattern(approved):
    query = str(approved.get('query') or '').strip()
    query = re.sub(r'^(Given|When|Then|And|But)\s+', '', query, flags=re.I)
    stage5 = approved.get('original_stage5_plan') or {}
    proposed = str(stage5.get('proposed_keyword_pattern') or '').strip()
    # Reuse Stage 5's validated pattern only if literal text matches the query.
    if proposed:
        parts = []
        cursor = 0
        for m in PLACEHOLDER.finditer(proposed):
            parts.append(re.escape(proposed[cursor:m.start()]))
            parts.append(r'(.+?)')
            cursor = m.end()
        parts.append(re.escape(proposed[cursor:]))
        if re.fullmatch(''.join(parts), query, flags=re.I):
            return proposed
    # Safe fallback: only quoted literal values become placeholders.
    matches = list(re.finditer(r'''(["'])(.*?)\1''', query))
    if not matches:
        return query
    pieces, pos = [], 0
    for i, m in enumerate(matches):
        pieces += [query[pos:m.start()], m.group(1) + '{' + ('value' if i == 0 else f'value_{i+1}') + '}' + m.group(1)]
        pos = m.end()
    return ''.join(pieces) + query[pos:]


def _tams(value):
    tokens = [x.strip() for x in str(value).split(',') if x.strip()]
    if not tokens or any(not re.fullmatch(r'\d+', t) for t in tokens):
        raise GenerationError('TAMS metadata must contain comma-separated integer IDs.')
    return [int(t) for t in tokens]


def _expression(value, placeholders):
    """Return Python expression; {placeholder} references Behave arguments."""
    value = str(value)
    tokens = list(PLACEHOLDER.finditer(value))
    if not tokens:
        return repr(value)
    for token in tokens:
        if token.group(1) not in placeholders:
            raise GenerationError(f'Unknown Behave placeholder {{{token.group(1)}}} in API parameter.')
    if len(tokens) == 1 and tokens[0].span() == (0, len(value)):
        return tokens[0].group(1)
    # f-string syntax from JSON ensures literal portions are escaped safely.
    chunks, last = [], 0
    for token in tokens:
        chunks.append(value[last:token.start()].replace('{','{{').replace('}','}}'))
        chunks.append('{' + token.group(1) + '}')
        last = token.end()
    chunks.append(value[last:].replace('{','{{').replace('}','}}'))
    return 'f' + repr(''.join(chunks))


def _pattern_segments(pattern):
    pos = 0
    for match in PLACEHOLDER.finditer(pattern):
        yield ('literal', pos, match.start())
        yield ('capture', match.start(), match.end())
        pos = match.end()
    yield ('literal', pos, len(pattern))


def keyword_placeholders(approved):
    """Return placeholders from the same resolved pattern used by generation."""
    return PLACEHOLDER.findall(_pattern(approved))


def _binding_expression(name, bindings, *, signal=False):
    config = bindings.get(name, {"mode": "direct"})
    mode = config.get("mode", "direct")
    if mode == "direct":
        if signal:
            raise GenerationError(f"Signal placeholder {{{name}}} requires an approved signal mapping.")
        return name
    if mode not in ("value_map", "signal_map"):
        raise GenerationError(f"Unsupported binding mode {mode!r} for {name}.")
    if (mode == "signal_map") != signal:
        raise GenerationError(f"Binding mode for {{{name}}} is incompatible with API field.")
    return f"_mapping_{name}[{name}.strip().lower()]"


def _check_bindings(bindings, names, signals):
    if not isinstance(bindings, dict):
        raise GenerationError("Parameter bindings must be a dictionary.")
    for name, cfg in bindings.items():
        if name not in names or not isinstance(cfg, dict):
            raise GenerationError(f"Unknown/invalid parameter binding: {name}")
        mode = cfg.get('mode', 'direct')
        if mode not in ('direct', 'value_map', 'signal_map'):
            raise GenerationError(f"Unsupported binding mode: {mode}")
        if mode == 'direct':
            continue
        mapping = cfg.get('mapping')
        if not isinstance(mapping, dict) or not mapping:
            raise GenerationError(f"Missing mapping for {{{name}}}.")
        normalized = [str(k).strip().lower() for k in mapping]
        if any(not x for x in normalized) or len(set(normalized)) != len(normalized):
            raise GenerationError(f"Empty or duplicate mapping inputs for {{{name}}}.")
        if any(str(v).strip() == '' for v in mapping.values()):
            raise GenerationError(f"Empty mapping output for {{{name}}}.")
        if mode == 'signal_map':
            missing = [v for v in mapping.values() if str(v) not in signals]
            if missing:
                raise GenerationError(f"Unknown signal keys in {{{name}}} mapping: {missing}")


def generate(approved, functions, signals, *, include_text_frame=False):
    if approved.get('decision') != 'APPROVED':
        raise GenerationError('Only an explicitly approved workflow can be generated.')
    if not approved.get('api_sequence'):
        raise GenerationError('Approved workflow has no API calls.')
    meta = approved.get('metadata') or {}
    platform = str(meta.get('platform') or '').strip()
    if not platform:
        raise GenerationError('Platform decorator metadata is missing.')
    tams = _tams(meta.get('tams',''))
    pattern = _pattern(approved)
    names = PLACEHOLDER.findall(pattern)
    if len(names) != len(set(names)) or any(not IDENT.fullmatch(n) or keyword.iskeyword(n) for n in names):
        raise GenerationError('Invalid or duplicate Behave pattern placeholders.')
    bindings = approved.get('parameter_bindings') or {}
    _check_bindings(bindings, names, signals)
    step_types = (approved.get('metadata') or {}).get('step_types') or ['given', 'when']
    if not isinstance(step_types, list) or not step_types or any(t not in ('given','when','then') for t in step_types):
        raise GenerationError('Select valid Gherkin decorators.')
    step_types = list(dict.fromkeys(step_types))
    for key in functions:
        if not isinstance(key,str):
            raise GenerationError('Invalid catalogue API key.')
    # Map approved literal values back to Behave captures when unambiguous.
    # This allows the existing UI to collect concrete values (30, 2) while
    # generated SDs remain reusable for other values of the same keyword.
    query = re.sub(r'^(Given|When|Then|And|But)\s+', '', str(approved.get('query') or '').strip(), flags=re.I)
    capture_regex = ''.join(
        re.escape(pattern[a:b]) if kind == 'literal' else r'(.+?)'
        for kind, a, b in _pattern_segments(pattern)
    )
    matched = re.fullmatch(capture_regex, query, flags=re.I)
    captured = {}
    if matched:
        for name, value in zip(names, matched.groups()):
            captured.setdefault(value, []).append(name)
    # Resolve UI literals against Gherkin captures. Prefer explicit {name}
    # references; allow percent-suffixed captures such as "50%" -> UI "50".
    # Never guess if multiple placeholders could match.
    aliases = {}
    for captured_value, possible_names in captured.items():
        for alias in (captured_value, captured_value[:-1].strip() if captured_value.endswith('%') else captured_value):
            aliases.setdefault(alias, set()).update(possible_names)
    bound_names = set()

    lines = [
        '"""Generated from a human-approved Stage 5.6 workflow. Review before execution."""',
        'from behave import ' + ', '.join(step_types),
        'from step_definitions_manager import TAMS, platform',
        '',
        f'@platform({platform!r})',
        f'@TAMS({tams!r})',
        *[f'@{step_type}({pattern!r})' for step_type in step_types],
        f'def step_impl(context{", " + ", ".join(names) if names else ""}):',
        '    """Build AutomationDesk library calls for this approved keyword."""',
        '    block_to_add = ""',
    ]
    for name in names:
        cfg = bindings.get(name, {})
        if cfg.get('mode') in ('value_map', 'signal_map'):
            mapping = {str(k).strip().lower(): str(v) for k,v in cfg['mapping'].items()}
            lines.append(f'    _mapping_{name} = {mapping!r}')
            lines.append(f'    if {name}.strip().lower() not in _mapping_{name}:')
            lines.append(f'        raise ValueError("Unsupported {name}: " + str({name}))')
    used = []
    if include_text_frame:
        if 'text_api' not in functions:
            raise GenerationError('Text Frame requested but text_api is absent from Function Catalogue.')
        raise GenerationError('Text Frame requires an approved parameter mapping; auto-insertion is disabled.')
    for i, step in enumerate(approved['api_sequence'], 1):
        api_key = step.get('api_key')
        if api_key not in functions:
            raise GenerationError(f'Step {i}: API {api_key!r} missing from Function Catalogue.')
        schema = functions[api_key]
        fields = schema.get('parameters') or {}
        if not fields:
            raise GenerationError(f'Step {i}: no structured parameter schema for {api_key}.')
        params = step.get('parameters') or {}
        if set(params) - set(fields):
            raise GenerationError(f'Step {i}: unrecognized parameters: {sorted(set(params)-set(fields))}')
        block_type = schema.get('block_type')
        if block_type not in ('Frame', 'Non Frame'):
            raise GenerationError(f'Step {i}: unsupported/missing block_type for {api_key}: {block_type!r}.')
        lines.append(f'    # API {i}: {api_key}')
        lines.append(f'    parameters_{i} = {{')
        groups = {}
        for full, field in fields.items():
            if '.' not in full:
                raise GenerationError(f'Step {i}: invalid field key {full!r}; expected Type.Name.')
            group, param = full.split('.', 1)
            if not IDENT.fullmatch(group) or not IDENT.fullmatch(param):
                raise GenerationError(f'Step {i}: invalid field name {full!r}.')
            value = params.get(full)
            if value is None or str(value).strip() == '':
                defaults = schema.get('default_values') or {}
                value = defaults.get(full, defaults.get(param))
            if value is None or str(value).strip() == '':
                raise GenerationError(f'Step {i}: missing parameter {full}.')
            direction = field.get('direction')
            if direction not in ('in','out','inout'):
                raise GenerationError(f'Step {i}: missing/unsupported direction for {full}.')
            groups.setdefault(group, []).append((param, value, direction, field))
            if param.lower() in ('variable','signal','signalname') or field.get('source') == 'Signal Catalogue':
                signal_placeholder = PLACEHOLDER.fullmatch(str(value))
                if signal_placeholder:
                    name = signal_placeholder.group(1)
                    if name not in names:
                        raise GenerationError(f'Unknown signal placeholder: {name}')
                    cfg = bindings.get(name, {})
                    if cfg.get('mode') != 'signal_map':
                        raise GenerationError(f'Signal {{{name}}} requires a verified signal mapping.')
                    for mapped_signal in cfg['mapping'].values():
                        if mapped_signal not in used:
                            used.append(mapped_signal)
                    bound_names.add(name)
                else:
                    if str(value) not in signals:
                        raise GenerationError(f'Step {i}: signal {value!r} not registered in Signal Catalogue.')
                    if str(value) not in used:
                        used.append(str(value))
        for group, entries in groups.items():
            lines.append(f'        {group!r}: {{')
            for param, value, direction, field in entries:
                if param.lower() in ('variable','signal','signalname') or field.get('source') == 'Signal Catalogue':
                    sig_match = PLACEHOLDER.fullmatch(str(value))
                    if sig_match:
                        expr = f'context.signal.calalogue.get_key_value({_binding_expression(sig_match.group(1), bindings, signal=True)})'
                    else:
                        expr = f'context.signal.calalogue.get_key_value({str(value)!r})'
                else:
                    literal = str(value)
                    if not PLACEHOLDER.search(literal):
                        candidates = aliases.get(literal, set())
                        if len(candidates) > 1:
                            raise GenerationError(
                                f'Step {i}: ambiguous parameter binding for {full}: {literal!r}. '
                                'Enter an explicit {placeholder} in the workflow parameter.'
                            )
                        if len(candidates) == 1:
                            literal = '{' + next(iter(candidates)) + '}'
                    mapped = PLACEHOLDER.fullmatch(literal)
                    if mapped and bindings.get(mapped.group(1), {}).get('mode') in ('value_map','signal_map'):
                        expr = _binding_expression(mapped.group(1), bindings)
                    else:
                        expr = _expression(literal, set(names))
                    bound_names.update(PLACEHOLDER.findall(literal))
                lines.append(f'            {param!r}: ["", {expr}, {direction!r}],')
            lines.append('        },')
        lines.append('    }')
        lines += [
            '    block_to_add += context.test_exec_gen.get_library_call(',
            f'        block_type={block_type!r},',
            f'        name={api_key!r} + "_{i}",',
            f'        lib_reference=context.api_calalogue.get_key_value({api_key!r}),',
            f'        arguments=parameters_{i},',
            '    )',
        ]
    unused = set(names) - bound_names
    if unused:
        raise GenerationError(
            'Gherkin placeholders not used by any API parameter: '
            + ', '.join(sorted(unused))
            + '. Bind each placeholder to an approved parameter before generating.'
        )
    lines.append('    context.xml_block_to_add = block_to_add')
    for signal in used:
        mapped_names = [n for n,cfg in bindings.items() if cfg.get('mode') == 'signal_map' and signal in cfg.get('mapping', {}).values()]
        if mapped_names:
            continue
        lines.append(f'    context.signals_used.append({signal!r})')
    for name,cfg in bindings.items():
        if cfg.get('mode') == 'signal_map':
            lines.append(f'    context.signals_used.append(_mapping_{name}[{name}.strip().lower()])')
    lines.append('')
    code = '\n'.join(lines)
    ast.parse(code)
    return {'code': code, 'pattern': pattern, 'api_count': len(approved['api_sequence']), 'signals_used': used,
            'warnings': ['Generated code passes syntax validation only; AutomationDesk runtime behavior is not verified.',
                         'Confirm API argument schemas and block types against authoritative ADPX metadata before production use.']}
