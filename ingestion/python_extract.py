from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DECORATOR_RE = re.compile(r"@(?P<name>\w+)\((?P<args>.*?)\)")
KEYWORD_RE = re.compile(r"@(given|when|then|step)\((?P<q>['\"])(?P<pattern>.*?)(?P=q)\)")
PLATFORM_RE = re.compile(r"@platform\((?P<q>['\"])(?P<value>.*?)(?P=q)\)")
TAMS_RE = re.compile(r"@TAMS\((?P<value>.*?)\)")
CATALOGUE_KEY_RE = re.compile(
    r"context\.(?P<kind>api_calalogue|api_catalogue|signal\.(?:calalogue|catalogue))\.get_key_value\(\s*['\"](?P<key>[^'\"]+)['\"]\s*\)"
)
APPEND_SIGNAL_RE = re.compile(
    r"context\.signals_used\.append\(\s*['\"](?P<key>[^'\"]+)['\"]\s*\)"
)
LIB_CALL_START_RE = re.compile(r"context\.test_exec_gen\.get_library_call\s*\(")


def safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<expression>"


def node_to_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return {"expression": node.id}
    if isinstance(node, ast.Attribute):
        return {"expression": safe_unparse(node)}
    if isinstance(node, ast.Subscript):
        return {"expression": safe_unparse(node)}
    if isinstance(node, ast.Call):
        return {"expression": safe_unparse(node)}
    if isinstance(node, ast.List):
        return [node_to_value(x) for x in node.elts]
    if isinstance(node, ast.Tuple):
        return [node_to_value(x) for x in node.elts]
    if isinstance(node, ast.Dict):
        out = {}
        for k, v in zip(node.keys, node.values):
            key = node_to_value(k)
            if isinstance(key, (dict, list)):
                key = safe_unparse(k)
            out[str(key)] = node_to_value(v)
        return out
    if isinstance(node, ast.UnaryOp):
        return {"expression": safe_unparse(node)}
    return {"expression": safe_unparse(node)}


def extract_docstring_metadata(source: str, module_docstring: Optional[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if module_docstring:
        result["module_docstring"] = module_docstring.strip()
        patterns = {
            "author": r"(?im)^\s*Author\s*[-:]\s*(.+)$",
            "domain": r"(?im)^\s*Domain\s*[-:]\s*(.+)$",
            "date_modified": r"(?im)^\s*date\s+Modified\s*[-:]\s*(.+)$",
        }
        for key, pattern in patterns.items():
            m = re.search(pattern, module_docstring)
            if m:
                result[key] = m.group(1).strip().strip('"')
    return result


def extract_literal_mappings(tree: ast.AST) -> Dict[str, Any]:
    mappings: Dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.endswith("_dict"):
                    mappings[target.id] = node_to_value(node.value)
    return mappings


def extract_context_outputs(tree: ast.AST) -> List[str]:
    outputs: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "context":
                    if target.attr not in outputs:
                        outputs.append(target.attr)
    return outputs


def extract_decorators_and_keyword(tree: ast.AST) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    metadata: Dict[str, Any] = {}
    keywords: List[Dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        metadata["function_name"] = node.name
        metadata["function_parameters"] = [a.arg for a in node.args.args if a.arg != "context"]
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            name = safe_unparse(dec.func)
            if name == "platform" and dec.args:
                value = node_to_value(dec.args[0])
                metadata["platform"] = value
            elif name == "TAMS" and dec.args:
                metadata["tams_ids"] = node_to_value(dec.args[0])
            elif name in {"given", "when", "then", "step", "step_definition"} and dec.args:
                pattern = node_to_value(dec.args[0])
                keywords.append({"step_type": name, "pattern": pattern})
    return metadata, keywords


def find_matching_paren(source: str, open_index: int) -> Optional[int]:
    depth = 0
    quote: Optional[str] = None
    escaped = False
    for i in range(open_index, len(source)):
        ch = source[i]
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def extract_library_call_nodes_from_source(source: str) -> List[Tuple[int, str]]:
    calls: List[Tuple[int, str]] = []
    for m in LIB_CALL_START_RE.finditer(source):
        end = find_matching_paren(source, m.end() - 1)
        if end is None:
            continue
        line = source.count("\n", 0, m.start()) + 1
        calls.append((line, source[m.start():end + 1]))
    return calls


def parse_library_call_expression(expr: str) -> Dict[str, Any]:
    try:
        tree = ast.parse("result = " + expr, mode="exec")
        assign = tree.body[0]
        call = assign.value
        if not isinstance(call, ast.Call):
            raise ValueError("not a call")
        data: Dict[str, Any] = {"call": "get_library_call"}
        for kw in call.keywords:
            if kw.arg in {"block_type", "name", "lib_reference"}:
                data[kw.arg] = node_to_value(kw.value)
            elif kw.arg == "arguments":
                data["arguments"] = node_to_value(kw.value)
            else:
                data[kw.arg] = node_to_value(kw.value)
        lib_ref = data.get("lib_reference")
        if isinstance(lib_ref, dict):
            expr_text = lib_ref.get("expression", "")
            m = re.search(r"get_key_value\(['\"]([^'\"]+)['\"]\)", expr_text)
            if m:
                data["api_key"] = m.group(1)
        return data
    except Exception as exc:
        return {"raw_expression": expr, "parse_error": str(exc)}


def extract_library_calls(tree: ast.AST, source: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get_library_call":
            if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "test_exec_gen":
                data = parse_library_call_expression(safe_unparse(node))
                data["line"] = getattr(node, "lineno", None)
                results.append(data)
    if results:
        return sorted(results, key=lambda x: x.get("line") or 0)
    for line, expr in extract_library_call_nodes_from_source(source):
        data = parse_library_call_expression(expr)
        data["line"] = line
        results.append(data)
    return results


def extract_catalogue_references(source: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    apis: List[Dict[str, Any]] = []
    signals: List[Dict[str, Any]] = []
    api_re = re.compile(r"context\.(?:api_calalogue|api_catalogue)\.get_key_value\(\s*['\"](?P<key>[^'\"]+)['\"]\s*\)")
    signal_re = re.compile(r"context\.signal\.(?:calalogue|catalogue)\.get_key_value\(\s*['\"](?P<key>[^'\"]+)['\"]\s*\)")
    for m in api_re.finditer(source):
        item = {"catalogue_key": m.group("key"), "kind": "api", "line": source.count("\n", 0, m.start()) + 1}
        if item["catalogue_key"] not in [x["catalogue_key"] for x in apis]:
            apis.append(item)
    for m in signal_re.finditer(source):
        item = {"catalogue_key": m.group("key"), "kind": "signal", "line": source.count("\n", 0, m.start()) + 1}
        if item["catalogue_key"] not in [x["catalogue_key"] for x in signals]:
            signals.append(item)
    for m in APPEND_SIGNAL_RE.finditer(source):
        key = m.group("key")
        if key not in [x["catalogue_key"] for x in signals]:
            signals.append({"catalogue_key": key, "kind": "signal", "usage": "signals_used.append", "line": source.count("\n", 0, m.start()) + 1})
    return apis, signals


def extract_keyword_fallback(source: str) -> List[Dict[str, Any]]:
    result = []
    for m in KEYWORD_RE.finditer(source):
        result.append({"step_type": m.group(1), "pattern": m.group("pattern")})
    return result


def extract_platform_tams_fallback(source: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    m = PLATFORM_RE.search(source)
    if m:
        result["platform"] = m.group("value")
    m = TAMS_RE.search(source)
    if m:
        try:
            result["tams_ids"] = ast.literal_eval(m.group("value"))
        except Exception:
            result["tams_ids"] = {"expression": m.group("value").strip()}
    m = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", source)
    if m:
        result["function_name"] = m.group(1)
        result["function_parameters"] = [x.strip() for x in m.group(2).split(",") if x.strip() and x.strip() != "context"]
    return result


def extract_value_mappings_fallback(source: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for m in re.finditer(r"(?P<name>\w+_dict)\s*=\s*\{(?P<body>.*?)\}", source, re.S):
        pairs = {}
        for pair in re.finditer(r"['\"]([^'\"]+)['\"]\s*:\s*['\"]([^'\"]+)['\"]", m.group("body")):
            pairs[pair.group(1)] = pair.group(2)
        if pairs:
            result[m.group("name")] = pairs
    return result


def extract_potential_observations(source: str, value_mappings: Dict[str, Any]) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    if re.search(r"\bclosed\b", source, re.I) and any("close" in str(v).lower() for v in value_mappings.values()):
        observations.append({
            "type": "potential_inconsistency",
            "code": "DOCSTRING_CLOSED_VS_CLOSE",
            "message": "Source text contains 'closed' while a value mapping uses 'close'; verify intended keyword value.",
        })
    for m in re.finditer(r"keyword\s*=\s*(['\"])(.*?)\1\s*\+\s*(\w+)", source):
        prefix = m.group(2)
        if not prefix.endswith(" "):
            observations.append({
                "type": "potential_inconsistency",
                "code": "KEYWORD_CONCATENATION_NO_SPACE",
                "message": "Keyword prefix is concatenated with a variable without a trailing space.",
                "prefix": prefix,
                "variable": m.group(3),
            })
    return observations


def parse_sd_file(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    result: Dict[str, Any] = {
        "sd_file": path.name,
        "source_path": str(path),
        "source_code": source,
        "parse_status": "ast",
        "syntax_error": None,
        "metadata": {},
        "keyword": {},
        "value_mappings": {},
        "library_calls": [],
        "catalogue_apis": [],
        "catalogue_signals": [],
        "context_outputs": [],
        "observations": [],
    }

    try:
        tree = ast.parse(source)
        module_docstring = ast.get_docstring(tree)
        result["metadata"].update(extract_docstring_metadata(source, module_docstring))
        dec_meta, keywords = extract_decorators_and_keyword(tree)
        result["metadata"].update(dec_meta)
        result["keyword"] = {"patterns": keywords}
        result["value_mappings"] = extract_literal_mappings(tree)
        result["library_calls"] = extract_library_calls(tree, source)
        result["context_outputs"] = extract_context_outputs(tree)
    except SyntaxError as exc:
        result["parse_status"] = "fallback"
        result["syntax_error"] = {
            "line": exc.lineno,
            "column": exc.offset,
            "message": exc.msg,
            "text": exc.text.strip() if exc.text else None,
        }
        result["metadata"].update(extract_docstring_metadata(source, source.split('"""', 2)[1] if '"""' in source else None))
        result["metadata"].update(extract_platform_tams_fallback(source))
        result["keyword"] = {"patterns": extract_keyword_fallback(source)}
        result["value_mappings"] = extract_value_mappings_fallback(source)
        result["library_calls"] = extract_library_calls_from_fallback(source)

    api_refs, signal_refs = extract_catalogue_references(source)
    result["catalogue_apis"] = api_refs
    result["catalogue_signals"] = signal_refs
    result["observations"] = extract_potential_observations(source, result["value_mappings"])
    return result



def find_matching_brace(source: str, open_index: int) -> Optional[int]:
    depth = 0
    quote: Optional[str] = None
    escaped = False
    for i in range(open_index, len(source)):
        ch = source[i]
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def extract_named_dict_assignments(source: str) -> List[Tuple[int, str, Dict[str, Any]]]:
    results: List[Tuple[int, str, Dict[str, Any]]] = []
    for m in re.finditer(r"(?m)^\s*(?P<name>parameters|\w+_dict)\s*=\s*\{", source):
        brace = source.find("{", m.start())
        end = find_matching_brace(source, brace)
        if end is None:
            continue
        expr = source[brace:end + 1]
        try:
            tree = ast.parse("result = " + expr, mode="exec")
            value = tree.body[0].value
            parsed = node_to_value(value)
            if isinstance(parsed, dict):
                results.append((source.count("\n", 0, m.start()) + 1, m.group("name"), parsed))
        except Exception:
            continue
    return results

def extract_library_calls_from_fallback(source: str) -> List[Dict[str, Any]]:
    results = []
    assignments = extract_named_dict_assignments(source)
    for line, expr in extract_library_call_nodes_from_source(source):
        data = parse_library_call_expression(expr)
        data["line"] = line
        previous = [x for x in assignments if x[0] < line and x[1] == "parameters"]
        if previous and ("arguments" not in data or data.get("arguments") == {"expression": "parameters"}):
            data["arguments"] = previous[-1][2]
            data["arguments_source_line"] = previous[-1][0]
        results.append(data)
    return results
