import copy
import streamlit as st

def init_state():
    defaults={"analysis_result":None,"human_answers":{},"decision":None,"approved_plan":None}
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=copy.deepcopy(v)

def reset_analysis():
    st.session_state.analysis_result=None
    st.session_state.human_answers={}
    st.session_state.decision=None
    st.session_state.approved_plan=None

def set_result(result):
    st.session_state.analysis_result=result
    st.session_state.human_answers={}
    st.session_state.decision=None
    st.session_state.approved_plan=None

def required_questions(result):
    return [q for q in (result.get("plan") or {}).get("questions",[]) if q.get("required")]

def all_required_answered(result):
    a=st.session_state.human_answers
    return all(str(a.get(q.get("id"),"")).strip() for q in required_questions(result))

def approved_envelope(result):
    questions=(result.get("plan") or {}).get("questions",[])
    answers=st.session_state.human_answers
    human=[]
    for q in questions:
        ans=str(answers.get(q.get("id"),"")).strip()
        if ans:
            human.append({"question_id":q.get("id"),"question":q.get("question"),
                          "answer":ans,"source":"human_confirmation",
                          "required":bool(q.get("required"))})
    return {"schema_version":"stage5.5-approved-plan-v1","decision":"APPROVED",
            "query":result.get("query"),"stage41_status":result.get("stage41_status"),
            "stage5_plan":copy.deepcopy(result.get("plan") or {}),
            "stage5_validation":copy.deepcopy(result.get("validation") or {}),
            "human_evidence":{"schema_version":"stage5.5-human-evidence-v1","items":human},
            "stage41_evidence":copy.deepcopy(result.get("stage41_evidence") or {}),
            "stage6_generation_authorized":False,
            "note":"Stage 6 remains locked in Stage 5.5."}
