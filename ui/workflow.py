import os
from retrieval.stage41 import Stage41Retriever
from retrieval.stage5 import NIMConfig,NIMPlannerClient,Stage5Planner

def _make_stage41():
    try:
        return Stage41Retriever()
    except TypeError as exc:
        raise RuntimeError(
          "Your Stage41Retriever requires constructor arguments. Copy the exact "
          "Stage 4/4.1 construction from the working stage5_planner.py into "
          "ui/workflow.py::_make_stage41(). Do not modify retrieval logic.") from exc

def _client():
    key=os.getenv("NVIDIA_API_KEY","").strip()
    if not key: raise RuntimeError("NVIDIA_API_KEY is not set.")
    cfg=NIMConfig(api_key=key,
        model=os.getenv("NVIDIA_MODEL","google/gemma-3-12b-it"),
        timeout=120,max_tokens=4096,temperature=0.1)
    return NIMPlannerClient(cfg)

def analyze_keyword(text,mode="nim",top_files=3,candidate_k=5):
    text=(text or "").strip()
    if not text: raise ValueError("Enter a Gherkin keyword.")
    planner=Stage5Planner(_make_stage41(),_client() if mode=="nim" else None)
    return planner.plan(text,mode=mode,top_files=top_files,candidate_k=candidate_k)

def extract_feature_steps(text):
    out=[]
    for raw in (text or "").splitlines():
        line=raw.strip()
        if not line or line.startswith("#"): continue
        for p in ("Given ","When ","Then ","And ","But "):
            if line.startswith(p):
                out.append(line); break
    return out

def analyze_feature(text,mode="nim",top_files=3,candidate_k=5):
    steps=extract_feature_steps(text)
    if not steps: raise ValueError("No executable Gherkin steps found.")
    return [analyze_keyword(s,mode,top_files,candidate_k) for s in steps]

from ui.replan import reconcile

def reevaluate_with_answers(result, answers):
    """Reconcile without changing Stage 4.1/5, or making another NIM call."""
    return reconcile(result, answers)
