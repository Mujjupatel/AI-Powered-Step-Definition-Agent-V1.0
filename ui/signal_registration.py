"""Local, engineer-confirmed signal catalogue overlay. Does not modify source catalogue."""
from __future__ import annotations
import json, os, re, tempfile
from pathlib import Path

KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TYPES = ("Boolean", "Float", "Integer", "Enum", "String")

def overlay_path(root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    return root / "data" / "approved_signals.json"

def read_overlay(root=None):
    path=overlay_path(root)
    if not path.exists(): return {}
    data=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data,dict): raise ValueError("Signal overlay must be a JSON object.")
    return data

def combined_signals(original,root=None):
    combined=dict(original)
    for key,entry in read_overlay(root).items():
        if key in combined and combined[key] != entry:
            raise ValueError(f"Signal key collision with source catalogue: {key}")
        combined[key]=entry
    return combined

def unknown_signals(steps,functions,signals):
    missing=set()
    for step in steps:
        schema=(functions.get(step.get("api_key")) or {}).get("parameters") or {}
        for name,meta in schema.items():
            short=name.split(".",1)[-1].lower()
            if short not in ("variable","signal","signalname") and meta.get("source")!="Signal Catalogue":
                continue
            value=str((step.get("parameters") or {}).get(name,"")).strip()
            if value and value not in signals: missing.add(value)
    return sorted(missing)

def validate_registration(entry,original,existing):
    key=str(entry.get("signal_key","")).strip()
    errors=[]
    if not KEY_RE.fullmatch(key): errors.append("Signal key must be a valid identifier.")
    if not str(entry.get("tams_id","")).strip(): errors.append("Signal TAMS ID is required.")
    if not str(entry.get("model_path","")).strip(): errors.append("Model path is required.")
    if entry.get("type") not in TYPES: errors.append("Choose a valid signal type.")
    if key in original: errors.append("Signal already exists in source catalogue.")
    if key in existing: errors.append("Signal already exists in approved overlay.")
    # We can check completeness, not whether a path exists on a physical HIL bench.
    return errors

def register_signal(entry,original,root=None):
    path=overlay_path(root)
    path.parent.mkdir(parents=True,exist_ok=True)
    existing=read_overlay(root)
    clean={k:str(entry.get(k,"")).strip() for k in
           ("signal_key","tams_id","model_path","type","description")}
    errors=validate_registration(clean,original,existing)
    if errors: raise ValueError("; ".join(errors))
    clean["source"]="engineer_confirmed_local_overlay"
    existing[clean["signal_key"]]=clean
    fd,tmp=tempfile.mkstemp(prefix=".approved_signals_",suffix=".tmp",dir=str(path.parent))
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(existing,f,indent=2,ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return clean
