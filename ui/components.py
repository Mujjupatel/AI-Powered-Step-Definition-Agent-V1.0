import streamlit as st
from ui.state import all_required_answered

def banner(r):
    s=r.get("stage41_status","UNKNOWN")
    if s=="EXACT_EXISTING": st.success("✅ EXACT EXISTING SD FILE AVAILABLE")
    elif s=="NEW_WITH_KNOWN_CATALOGUE": st.warning("🟠 THIS IS A NEW KEYWORD — BUT SIGNALS AVAILABLE")
    else: st.error("⚠️ NO EXACT MATCH FOUND — THIS IS A NEW KEYWORD")

def plan_view(r):
    p=r.get("plan") or {}; st.subheader("Generation plan"); st.write(p.get("summary",""))
    a,b,c=st.columns(3); a.metric("Step type",str(p.get("step_type") or "—").upper())
    b.metric("Signals",len(p.get("selected_signals") or [])); c.metric("APIs",len(p.get("proposed_apis") or []))
    if p.get("proposed_keyword_pattern"): st.code(p["proposed_keyword_pattern"],language=None)

def evidence_view(r):
    e=r.get("stage41_evidence") or {}; st.subheader("Evidence")
    l,rr=st.columns(2)
    with l:
        st.markdown("**Confirmed signals**")
        for s in e.get("known_signals") or []:
            key=s.get("signal_key") or s.get("Signal Key")
            st.success(key); st.caption(f'{s.get("Type","—")} · {s.get("TAMS ID","—")} · {s.get("Model Path","—")}')
        if not e.get("known_signals"): st.info("No confirmed signal.")
    with rr:
        st.markdown("**APIs**")
        for a in e.get("confirmed_apis") or []: st.success("Confirmed · "+a.get("api_key",""))
        for a in e.get("candidate_apis") or []: st.warning("Candidate · "+a.get("api_key",""))
        if not e.get("confirmed_apis") and not e.get("candidate_apis"): st.info("No confirmed/candidate API.")

def parameter_view(r):
    rows=[]
    for x in (r.get("plan") or {}).get("parameter_plan") or []:
        for k,v in (x.get("parameters") or {}).items(): rows.append({"API":x.get("api_key"),"Parameter":k,"Value":v})
    if rows: st.subheader("Parameter plan"); st.dataframe(rows,use_container_width=True,hide_index=True)

def questions_view(r):
    qs=(r.get("plan") or {}).get("questions") or []
    if not qs:return
    st.subheader("Information required")
    st.caption("Answers are stored as human-confirmed evidence; retrieval evidence is not overwritten.")
    for q in qs:
        qid=str(q.get("id")); label=q.get("question",qid)+(" *" if q.get("required") else "")
        st.session_state.human_answers[qid]=st.text_input(label,value=str(st.session_state.human_answers.get(qid,"")),
                                                          key="answer_"+qid,help=q.get("reason"))
    if not all_required_answered(r): st.info("Complete all required fields.")

def extras(r):
    v=r.get("validation") or {}
    if v.get("valid"): st.success("Stage 5 validation passed · closed-list guardrails enforced.")
    else: st.error("Stage 5 validation failed: "+str(v.get("errors",[])))
    with st.expander("Similar SD evidence"):
        for x in (r.get("stage41_evidence") or {}).get("similar_sd_files") or []:
            st.write(f'{x.get("sd_file")} · score {x.get("score")} · authority={x.get("implementation_authority")}')
    with st.expander("Technical details / raw JSON"): st.json(r)
