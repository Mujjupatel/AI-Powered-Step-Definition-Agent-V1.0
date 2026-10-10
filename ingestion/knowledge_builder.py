from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .models import Issue, KnowledgeCard, KnowledgeChunk
from .python_extract import parse_sd_file
from .feature_parser import parse_feature_file


def stable_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _issue(severity: str, code: str, message: str, file: str, line=None, **details) -> Issue:
    return Issue(severity=severity, code=code, message=message, file=file, line=line, details=details)


def validate_sd(parsed: Dict[str, Any], functions: Dict[str, Dict[str, Any]], signals: Dict[str, Dict[str, Any]]) -> List[Issue]:
    issues: List[Issue] = []
    filename = parsed["sd_file"]
    if parsed.get("syntax_error"):
        err = parsed["syntax_error"]
        issues.append(_issue("ERROR", "SYNTAX_ERROR", f"Python syntax error: {err.get('message')}", filename, err.get("line"), syntax_error=err))

    seen_api_keys = {x["catalogue_key"] for x in parsed.get("catalogue_apis", [])}
    for ref in parsed.get("catalogue_apis", []):
        key = ref.get("catalogue_key")
        if key and key not in functions:
            issues.append(_issue("ERROR", "UNKNOWN_API", f"API '{key}' is not present in the Function Catalogue.", filename, ref.get("line"), api_key=key))

    for call in parsed.get("library_calls", []):
        key = call.get("api_key")
        if key and key not in seen_api_keys:
            parsed.setdefault("catalogue_apis", []).append({"catalogue_key": key, "kind": "api", "usage": "get_library_call"})
            seen_api_keys.add(key)

        if key and key not in functions:
            issues.append(_issue("ERROR", "UNKNOWN_API", f"API '{key}' is not present in the Function Catalogue.", filename, call.get("line"), api_key=key))
            continue

        if key in functions:
            actual = call.get("arguments", {})
            if isinstance(actual, dict):
                expected_schema = functions[key].get("parameters", {})
                expected_names = {str(k).split(".", 1)[-1] for k in expected_schema}
                actual_names = set()
                for type_group, params in actual.items():
                    if isinstance(params, dict):
                        actual_names.update(params.keys())
                unknown_params = sorted(actual_names - expected_names)
                if unknown_params:
                    issues.append(_issue("ERROR", "UNKNOWN_API_PARAMETER", f"API '{key}' uses parameters not defined in the catalogue: {unknown_params}.", filename, call.get("line"), api_key=key, parameters=unknown_params))

    for ref in parsed.get("catalogue_signals", []):
        key = ref.get("catalogue_key")
        if key and key not in signals:
            issues.append(_issue("ERROR", "UNKNOWN_SIGNAL", f"Signal '{key}' is not present in the Signal Catalogue.", filename, ref.get("line"), signal_key=key))

    for obs in parsed.get("observations", []):
        issues.append(_issue("WARNING", obs.get("code", "OBSERVATION"), obs.get("message", "Source observation."), filename, details={k: v for k, v in obs.items() if k not in {"type", "code", "message"}}))

    if not parsed.get("keyword", {}).get("patterns"):
        issues.append(_issue("WARNING", "NO_KEYWORD_PATTERN", "No Behave keyword decorator was extracted.", filename))
    if not parsed.get("library_calls"):
        issues.append(_issue("WARNING", "NO_LIBRARY_CALL", "No get_library_call usage was extracted.", filename))
    return issues


def build_knowledge_card(parsed: Dict[str, Any], issues: List[Issue]) -> KnowledgeCard:
    calls = parsed.get("library_calls", [])
    api_items = []
    api_parameter_mapping: Dict[str, Any] = {}
    for call in calls:
        api_key = call.get("api_key")
        if not api_key:
            continue
        item = {
            "catalogue_key": api_key,
            "block_type": call.get("block_type"),
            "library_name": call.get("name"),
            "line": call.get("line"),
        }
        if item not in api_items:
            api_items.append(item)
        api_parameter_mapping.setdefault(api_key, []).append(call.get("arguments", {}))

    signal_items = []
    for ref in parsed.get("catalogue_signals", []):
        item = {k: v for k, v in ref.items() if k != "kind"}
        if item not in signal_items:
            signal_items.append(item)

    sequence = []
    for call in calls:
        sequence.append({"type": "api_call", "api_key": call.get("api_key"), "line": call.get("line")})
    for signal in signal_items:
        if signal.get("usage") == "signals_used.append":
            sequence.append({"type": "signal_usage", "signal_key": signal.get("catalogue_key"), "line": signal.get("line")})
    for output in parsed.get("context_outputs", []):
        sequence.append({"type": "context_output", "name": output})

    keyword_patterns = parsed.get("keyword", {}).get("patterns", [])
    keyword_params = parsed.get("metadata", {}).get("function_parameters", [])

    return KnowledgeCard(
        sd_file=parsed["sd_file"],
        source_path=parsed["source_path"],
        source_code=parsed["source_code"],
        parse_status=parsed["parse_status"],
        syntax_error=parsed.get("syntax_error"),
        metadata=parsed.get("metadata", {}),
        keyword={"patterns": keyword_patterns, "parameters": keyword_params},
        value_mappings=parsed.get("value_mappings", {}),
        apis=api_items,
        signals=signal_items,
        api_parameter_mapping=api_parameter_mapping,
        execution_sequence=sequence,
        context_outputs=parsed.get("context_outputs", []),
        observations=parsed.get("observations", []),
        issues=issues,
    )


def card_to_chunks(card: KnowledgeCard) -> List[KnowledgeChunk]:
    chunks: List[KnowledgeChunk] = []
    base = card.sd_file
    meta = {"sd_file": base, "domain": card.metadata.get("domain"), "platform": card.metadata.get("platform"), "parse_status": card.parse_status}

    for p in card.keyword.get("patterns", []):
        pattern = p.get("pattern")
        step_type = p.get("step_type")
        text = f"Keyword pattern: {pattern}\nStep type: {step_type}\nParameters: {', '.join(card.keyword.get('parameters', [])) or 'none'}"
        chunks.append(KnowledgeChunk(stable_id(base, "keyword", str(pattern)), base, "keyword_pattern", text, {**meta, "step_type": step_type, "keyword_pattern": pattern}))

    for api in card.apis:
        api_key = api.get("catalogue_key")
        mapping = card.api_parameter_mapping.get(api_key, [])
        text = f"API usage: {api_key}\nBlock type: {api.get('block_type')}\nLibrary name: {api.get('library_name')}\nObserved parameter mapping: {json.dumps(mapping, ensure_ascii=False)}"
        chunks.append(KnowledgeChunk(stable_id(base, "api", str(api_key)), base, "api_usage", text, {**meta, "api_key": api_key, "block_type": api.get("block_type")}))

    for signal in card.signals:
        key = signal.get("catalogue_key")
        text = f"Signal usage: {key}\nUsage: {signal.get('usage', 'catalogue reference')}\nObserved in SD: {base}"
        chunks.append(KnowledgeChunk(stable_id(base, "signal", str(key)), base, "signal_usage", text, {**meta, "signal_key": key}))

    if card.value_mappings:
        text = f"Value mappings in {base}:\n{json.dumps(card.value_mappings, indent=2, ensure_ascii=False)}"
        chunks.append(KnowledgeChunk(stable_id(base, "mapping"), base, "value_mapping", text, {**meta, "mapping_names": list(card.value_mappings)}))

    if card.execution_sequence:
        text = f"Execution sequence in {base}:\n" + "\n".join(json.dumps(x, ensure_ascii=False) for x in card.execution_sequence)
        chunks.append(KnowledgeChunk(stable_id(base, "flow"), base, "execution_flow", text, meta))

    if card.api_parameter_mapping:
        text = f"Parameter mapping in {base}:\n{json.dumps(card.api_parameter_mapping, indent=2, ensure_ascii=False)}"
        chunks.append(KnowledgeChunk(stable_id(base, "parameters"), base, "parameter_mapping", text, meta))

    chunks.append(KnowledgeChunk(stable_id(base, "source"), base, "source_code", card.source_code, {**meta, "source_kind": "raw_python"}))

    if card.observations:
        text = f"Observed caveats in {base}:\n" + "\n".join(o.get("message", "") for o in card.observations)
        chunks.append(KnowledgeChunk(stable_id(base, "observations"), base, "observations", text, meta))
    return chunks


def build_feature_chunks(features: List[Dict[str, Any]]) -> List[KnowledgeChunk]:
    chunks = []
    for feature in features:
        for scenario in feature.get("scenarios", []):
            text = f"Feature: {feature.get('feature')}\nScenario: {scenario.get('name')}\n" + "\n".join(f"{s['keyword']} {s['text']}" for s in scenario.get("steps", []))
            chunks.append(KnowledgeChunk(stable_id(feature["feature_file"], scenario["name"]), feature["feature_file"], "gherkin_scenario", text, {"feature": feature.get("feature"), "scenario": scenario.get("name")}))
    return chunks
