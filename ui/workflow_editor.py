"""Stage 5.5 catalogue-backed ordered API workflow, independent of Stage 5."""
from __future__ import annotations
import copy
import json
import re
from pathlib import Path

def load_catalogue(project_root=None):
    root = Path(project_root or Path(__file__).resolve().parents[1])
    path = root / "example_output" / "catalogue_snapshot.json"
    if not path.exists():
        raise FileNotFoundError(f"Catalogue snapshot missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    funcs = raw.get("functions", {})
    signals = raw.get("signals", {})
    if isinstance(funcs, list):
        funcs = {x["api_key"]: x for x in funcs}
    if isinstance(signals, list):
        signals = {x.get("signal_key"): x for x in signals}
    return funcs, signals

def api_frequency(project_root=None):
    """Count API mentions in existing extracted SD knowledge; do not invent popularity."""
    root = Path(project_root or Path(__file__).resolve().parents[1])
    files = list((root / "example_output").glob("knowledge_cards.json"))
    counts = {}
    if files:
        try:
            cards = json.loads(files[0].read_text(encoding="utf-8"))
            records = cards if isinstance(cards, list) else cards.get("sd_files", cards.get("cards", []))
            for item in records:
                for api in item.get("api_usage", item.get("api_calls", [])):
                    key = api if isinstance(api, str) else api.get("api_key")
                    if key: counts[key] = counts.get(key, 0) + 1
        except (ValueError, TypeError, KeyError):
            pass
    return counts

def recommendations(result, funcs):
    """Use only catalogue API keys; NIM candidate first, intent fallback second."""
    plan = result.get("plan") or {}
    evidence = result.get("stage41_evidence") or {}
    proposed = [x.get("api_key") for x in plan.get("proposed_apis", []) if isinstance(x,dict)]
    proposed += [x.get("api_key") for x in evidence.get("candidate_apis", []) if isinstance(x,dict)]
    proposed += [x.get("api_key") for x in evidence.get("confirmed_apis", []) if isinstance(x,dict)]
    query = str(result.get("query") or "").lower()
    # Heuristic is an optional proposal, not a mandatory workflow.
    operations = []
    if re.search(r"\b(set|write|apply|change|open|close|activate|shift)\b",query):
        operations.append(("set_signal","Set requested signal"))
    if re.search(r"\b(wait|sleep|delay|after|seconds?)\b",query):
        operations.append(("wait","Wait for settling time"))
    if re.search(r"\b(check|verify|validate|should|assert|confirm|verified)\b",query):
        operations.append(("check_signal","Verify expected signal"))
    sequence=[]
    for key,why in operations:
        if key in funcs and key not in [x["api_key"] for x in sequence]:
            sequence.append({"api_key":key,"reason":why,"source":"intent_heuristic"})
    if not sequence:
        for key in proposed:
            if key in funcs and key not in [x["api_key"] for x in sequence]:
                sequence.append({"api_key":key,"reason":"Stage 5 / catalogue recommendation","source":"stage5_candidate"})
    return sequence

def default_params(api, result):
    """Defaults are POC hints; missing required values remain blank."""
    defaults = api.get("default_values") or {}
    existing = {}
    for x in (result.get("plan") or {}).get("parameter_plan", []):
        if x.get("api_key") == api.get("api_key"):
            existing.update(x.get("parameters") or {})
    params = {}
    for full,meta in (api.get("parameters") or {}).items():
        short = full.split(".",1)[-1]
        val = existing.get(full, existing.get(short, defaults.get(short, defaults.get(full,""))))
        params[full] = str(val) if val is not None else ""
    return params

def validate_workflow(result, steps, funcs, signals, metadata=None):
    """Validate selected API existence, typed parameters, required values, signal keys.
    Human selection is not treated as catalogue-authoritative until lookup succeeds.
    """
    errors=[]
    if not steps: errors.append("At least one API step is required.")
    checked=[]
    known_keys = set(signals)
    evidence_keys = {s.get("signal_key") for s in (result.get("stage41_evidence") or {}).get("known_signals",[]) if isinstance(s,dict)}
    for i,step in enumerate(steps,1):
        key=step.get("api_key")
        if key not in funcs:
            errors.append(f"Step {i}: API '{key}' is not in Function Catalogue.")
            continue
        schema=funcs[key]
        params=step.get("parameters") or {}
        declared=schema.get("parameters") or {}
        for name in params:
            if declared and name not in declared:
                errors.append(f"Step {i}: unknown parameter {name} for {key}.")
        if not declared:
            errors.append(f"Step {i}: parameter schema unavailable for {key}; cannot validate.")
        for full,meta in declared.items():
            short=full.split(".",1)[-1]
            value=str(params.get(full,"")).strip()
            if not value and short not in (schema.get("default_values") or {}):
                errors.append(f"Step {i}: missing {full}.")
            if value and (short.lower() in ("variable","signal","signalname") or meta.get("source")=="Signal Catalogue"):
                if value not in known_keys:
                    errors.append(f"Step {i}: signal '{value}' not found in Signal Catalogue.")
            if value and meta.get("type","").lower() in ("float","integer","int"):
                try:
                    float(value)
                except ValueError:
                    errors.append(f"Step {i}: {full} must be numeric.")
        checked.append({"order":i,"api_key":key,"parameters":copy.deepcopy(params),
                        "source":"catalogue_verified_human_selection","reason":step.get("reason","")})
    meta = metadata or {}
    if not str(meta.get("platform","")).strip(): errors.append("Platform metadata required.")
    if not str(meta.get("tams","")).strip(): errors.append("TAMS metadata required.")
    return {"valid":not errors,"errors":errors,"api_sequence":checked,
            "metadata":copy.deepcopy(meta),"stage6_generation_authorized":False,
            "schema_version":"stage5.5-workflow-v1",
            "signal_catalogue_keys_checked":sorted(known_keys),
            "source_query":result.get("query")}

def approved_workflow(result, validation):
    if not validation.get("valid"): raise ValueError("Workflow validation failed.")
    return {"schema_version":"stage5.5-approved-workflow-v1",
            "query":result.get("query"),"decision":"APPROVED",
            "api_sequence":copy.deepcopy(validation["api_sequence"]),
            "metadata":copy.deepcopy(validation["metadata"]),
            "stage41_status":result.get("stage41_status"),
            "original_stage5_plan":copy.deepcopy(result.get("plan")),
            "stage6_generation_authorized":False}
