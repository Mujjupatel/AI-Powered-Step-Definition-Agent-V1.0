"""Stage 7.2 conversational workspace. Run alongside the existing streamlit_app.py."""
import copy
import ast
import hashlib
import json
import os
import streamlit as st

from ui.workflow import analyze_keyword, extract_feature_steps, analyze_feature
from ui.replan import reconcile
from ui.components import banner, plan_view, evidence_view, parameter_view, extras
from ui.workflow_editor import api_frequency
from ui.ai_workflow_planner import plan_workflow
from ui.workflow_editor import load_catalogue, recommendations, default_params, validate_workflow, approved_workflow
from ui.signal_registration import combined_signals, unknown_signals, register_signal, TYPES
from ui.stage6_generator import generate, GenerationError, keyword_placeholders, _check_bindings
from ui.evidence_reuse import propose_reuse, apply_reuse_to_steps
from ui.sd_merge import merge_existing_sd, MergeError

st.set_page_config(page_title="SD Agent · Workspace", page_icon="🤖", layout="wide")
st.markdown("""<style>
.block-container{max-width:1150px;padding-top:1.7rem}
.stChatMessage{border:1px solid rgba(128,145,180,.15);border-radius:16px}
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px}
</style>""", unsafe_allow_html=True)

def init():
    defaults = {"s72_messages":[{"role":"assistant","content":"Welcome! Paste a Gherkin keyword or upload a feature file. I'll help you build, validate and approve an AutomationDesk step definition."}],
                "s72_result":None,"s72_steps":[],"s72_validation":None,"s72_approved":None,
                "s72_generated":None,"s72_signature":None,"s72_ai":None,"s72_nextid":1,
                "s72_types":[],"s72_platform":"","s72_tams":"","s72_mode":"nim",
                "s72_selected_feature":None,"s72_bindings":{},
                "s72_top_files":3,"s72_candidate_k":5,"s72_feature_results":[],
                "s72_answers":{},"s72_replanned":None,"s72_merged":None,"s72_original_source":None,"s72_original_filename":None}
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=copy.deepcopy(v)

def say(role,content):
    st.session_state.s72_messages.append({"role":role,"content":content})

def invalidate():
    st.session_state.s72_validation=None
    st.session_state.s72_approved=None
    st.session_state.s72_generated=None
    st.session_state.s72_merged=None
    st.session_state.s72_signature=None

def current_signature(steps,signals):
    payload={"query":(st.session_state.s72_result or {}).get("query"),"steps":steps,
             "platform":st.session_state.s72_platform,"tams":st.session_state.s72_tams,
             "step_types":st.session_state.s72_types,"signals":signals,
             "bindings":st.session_state.s72_bindings}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()

def clear_editor_widgets():
    for key in list(st.session_state.keys()):
        if key.startswith(("s72_api_", "s72_p_", "s72_map_", "s72_bind_")):
            del st.session_state[key]

def existing_sd_source(result, filename):
    """Return source only if it is present in retrieved evidence; never fabricate code."""
    evidence=(result.get("stage41_evidence") or {}).get("retrieval_evidence") or {}
    for item in evidence.get("rag_context") or []:
        if item.get("sd_file") != filename:
            continue
        for source in item.get("source_code") or []:
            if isinstance(source,str) and source.strip():
                return source
    return None


def friendly_status(result):
    status=result.get("stage41_status", "")
    evidence=result.get("stage41_evidence") or {}
    extension=evidence.get("existing_pattern_extension")
    match=evidence.get("best_matching_keyword") or {}
    if extension:
        return ("Existing Step Definition found — requested decorator is missing",
                f"The keyword pattern is available in **{extension['sd_file']}**, "
                f"but **@{extension['requested_step_type']}** is not implemented. "
                "Would you like to review extending this SD file? No changes will be made without your approval.")
    if status=="EXACT_EXISTING":
        filename=match.get("sd_file") or "the indexed Step Definition repository"
        return ("Step Definition already available",
                f"Your requested keyword and step type are already implemented in **{filename}**. "
                "Review the existing SD below; you do not need to generate a duplicate.")
    if status=="NEW_WITH_KNOWN_CATALOGUE":
        return ("New Step Definition needed — catalogue information found",
                "No existing implementation matched this keyword and decorator. "
                "Relevant catalogue entries are available for planning; please review the proposed workflow.")
    return ("No matching Step Definition confirmed",
            "I couldn't confirm an existing implementation for this keyword. "
            "Review the retrieved evidence and provide any missing signal or API information.")


def analyze(query,funcs):
    query=query.strip()
    if not query: return
    result=analyze_keyword(query,st.session_state.s72_mode,st.session_state.s72_top_files,st.session_state.s72_candidate_k)
    clear_editor_widgets()
    st.session_state.s72_result=result
    st.session_state.s72_ai=None
    st.session_state.s72_merged=None
    st.session_state.s72_replanned=None
    st.session_state.s72_answers={}
    st.session_state.s72_bindings={}
    st.session_state.s72_reuse=None
    extension=(result.get("stage41_evidence") or {}).get("existing_pattern_extension")
    proposed=recommendations(result,funcs)
    if extension and extension.get("requested_step_type")=="then":
        # Never propose the existing setter for a Then verification.
        proposed=[x for x in proposed if x.get("api_key") not in
                  {"set_signal","set_enum_signal","toggle_signal","set_multiple_signals","ramp_signal"}]
        if not proposed:
            proposed=[{"api_key":key,"reason":"Candidate verification API; engineer must validate signal and expected value"}
                      for key in ("check_signal","compare_signal") if key in funcs][:1]
    st.session_state.s72_steps=[
        {"id":i+1,"api_key":x["api_key"],"reason":x["reason"],
         "parameters":default_params(funcs[x["api_key"]],result)}
        for i,x in enumerate(proposed) if x["api_key"] in funcs
    ]
    proposal=propose_reuse(result,funcs,combined_signals(dict(load_catalogue()[1])))
    st.session_state.s72_reuse=proposal
    if proposal:
        result.setdefault('plan',{})['proposed_keyword_pattern']=proposal['pattern']
    if proposal and proposal['mapping']:
        st.session_state.s72_bindings[proposal['placeholder']]={"mode":"value_map","mapping":proposal['mapping']}
    apply_reuse_to_steps(st.session_state.s72_steps,proposal,funcs)
    st.session_state.s72_nextid=len(st.session_state.s72_steps)+1
    invalidate()
    say("user",query)
    heading, explanation=friendly_status(result)
    say("assistant",f"**{heading}**\n\n{explanation}")

init()
with st.sidebar:
    st.title("🤖 SD Agent")
    st.caption("Conversational workspace · Stage 7.2")
    st.session_state.s72_mode=st.selectbox("Planner mode",["nim","offline"],index=["nim","offline"].index(st.session_state.s72_mode))
    st.caption("Model: "+os.getenv("NVIDIA_MODEL","configured model"))
    st.slider("Similar SD files",1,6,key="s72_top_files")
    st.slider("Candidate retrieval depth",1,10,key="s72_candidate_k")
    if st.button("＋ New session",use_container_width=True):
        for k in list(st.session_state):
            if k.startswith("s72_"): del st.session_state[k]
        st.rerun()
    st.divider()
    st.markdown("**Workflow progress**")
    for label,done in [
        ("Keyword analyzed",bool(st.session_state.s72_result)),
        ("Workflow validated",bool(st.session_state.s72_validation and st.session_state.s72_validation.get("valid"))),
        ("Engineer approved",bool(st.session_state.s72_approved)),
        ("Python generated",bool(st.session_state.s72_generated))]:
        st.write(("✅ " if done else "○ ")+label)
    st.divider()
    st.caption("Stage 7.2 consolidated workspace")

st.title("Step Definition Agent")
st.caption("From Gherkin to human-approved AutomationDesk Python · catalogue-validated")
try:
    funcs,base_signals=load_catalogue()
    signals=combined_signals(dict(base_signals))
except Exception as exc:
    st.error(f"Cannot load catalogues: {exc}")
    st.stop()

for m in st.session_state.s72_messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

with st.expander("📎 Upload or paste a feature file",expanded=False):
    uploaded=st.file_uploader("Feature file",type=["feature","txt"],key="s72_file")
    pasted=st.text_area("Or paste feature content",key="s72_feature_text",height=120)
    feature=(uploaded.getvalue().decode("utf-8",errors="replace") if uploaded else pasted)
    if feature:
        extracted=extract_feature_steps(feature)
        if extracted:
            selected=st.selectbox("Select a step to analyze",extracted,key="s72_feature_choice")
            if st.button("Analyze selected feature step"):
                try:
                    with st.spinner("Retrieving evidence and planning..."): analyze(selected,funcs)
                    st.rerun()
                except Exception as exc: st.error(str(exc))
            if st.button("Analyze all feature steps"):
                try:
                    with st.spinner("Analyzing each feature step..."):
                        st.session_state.s72_feature_results=analyze_feature(
                            feature,st.session_state.s72_mode,
                            st.session_state.s72_top_files,st.session_state.s72_candidate_k)
                    st.rerun()
                except Exception as exc: st.error(str(exc))
        else: st.warning("No executable Gherkin steps found.")
    if st.session_state.s72_feature_results:
        st.caption(f"Analyzed {len(st.session_state.s72_feature_results)} feature steps")
        for i,fr in enumerate(st.session_state.s72_feature_results):
            with st.expander(f"{i+1}. {fr.get('query','')} · {friendly_status(fr)[0]}"):
                plan_view(fr)
                extras(fr)
                if st.button("Open in workflow editor",key=f"s72_open_feature_{i}"):
                    analyze(fr.get('query',''),funcs)
                    st.rerun()

query=st.chat_input("Enter a Gherkin keyword (e.g. the window position is set to \"50%\")")
if query:
    try:
        with st.spinner("Analyzing with RAG and Stage 5..."): analyze(query,funcs)
        st.rerun()
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

r=st.session_state.s72_result
if r:
    with st.container(border=True):
        st.subheader("1 · Analysis")
        st.write("**Keyword:**",r.get("query",""))
        heading, explanation=friendly_status(r)
        st.markdown(f"**{heading}**")
        st.info(explanation)
        evidence=r.get("stage41_evidence") or {}
        extension=evidence.get("existing_pattern_extension")
        match=evidence.get("best_matching_keyword") or {}
        sd_file=(extension or match).get("sd_file")
        if sd_file:
            st.write("**Existing SD file:**",sd_file)
            st.write("**Matched keyword pattern:**",(extension or match).get("pattern","Not available"))
            if extension:
                st.write("**Available decorators:**",", ".join(extension.get("available_step_types") or []))
                st.write("**Requested decorator:**",extension.get("requested_step_type",""))
                st.write("**Existing SD APIs:**",", ".join(extension.get("observed_apis") or []) or "Not identified")
                st.write("**Existing SD signals:**",", ".join(extension.get("observed_signals") or []) or "Not identified")
                st.warning("Adding a decorator to a signal-setting function may be unsafe for Then verification. "
                           "Review the proposed separate implementation before approving any code.")
            with st.expander(f"View existing SD source — {sd_file}"):
                source=existing_sd_source(r,sd_file)
                if source:
                    st.code(source,language="python")
                else:
                    st.caption("Full source is not available in the retrieved evidence. "
                               "Open the named file in your project to review its implementation.")
        if extension:
            st.caption("This workspace prepares a reviewable workflow; it does not edit the original SD in place.")
            st.markdown("**Original SD for safe merge (required)**")
            original_upload=st.file_uploader(
                "Upload the complete existing Python SD file (not a RAG excerpt)",
                type=["py"],key="s72_original_upload")
            if original_upload is not None:
                if original_upload.name != extension.get("sd_file"):
                    st.error(f"Expected {extension.get('sd_file')}; received {original_upload.name}.")
                    st.session_state.s72_original_source=None
                    st.session_state.s72_original_filename=None
                else:
                    candidate=original_upload.getvalue().decode("utf-8-sig",errors="strict")
                    try:
                        ast.parse(candidate,filename=original_upload.name)
                    except SyntaxError as exc:
                        st.error(f"Original SD is not complete valid Python: line {exc.lineno}: {exc.msg}")
                        st.session_state.s72_original_source=None
                        st.session_state.s72_original_filename=None
                    else:
                        st.session_state.s72_original_source=candidate
                        st.session_state.s72_original_filename=original_upload.name
                        st.success("Complete original SD accepted for merge.")
        with st.expander("Evidence and Stage 5 plan"):
            st.json({"stage41_evidence":r.get("stage41_evidence"),"plan":r.get("plan"),
                     "validation":r.get("validation")})
        with st.expander("Detailed evidence and parameter plan"):
            evidence_view(r)
            parameter_view(r)
            extras(r)
        reuse=st.session_state.s72_reuse
        if reuse:
            with st.expander("Stage 7.3 · Existing SD reuse suggestions",expanded=True):
                st.write("**Original file:**",reuse['source_file'])
                st.write("**Canonical pattern:**",reuse['pattern'])
                st.write("**Catalogue-confirmed signal:**",reuse['signal'] or "Not confirmed; select a verified signal")
                st.write("**Verification API candidate:**",reuse['api'] or "Not available")
                st.write("**Value mapping proposal:**",reuse['mapping'] or "Not extractable from full source")
                st.warning("These are proposals, not automatic approval. Confirm mapping, signal, API and bench compatibility before validation.")
        questions=(r.get("plan") or {}).get("questions") or []
        if reuse and reuse.get('api'):
            questions=[q for q in questions if not __import__('re').search(r'which api.*(set or verify|set the)',q.get('question',''),__import__('re').I)]
        if questions:
            st.markdown("**Stage 5 clarification questions**")
            for q in questions:
                qid=str(q.get("id",""))
                st.session_state.s72_answers[qid]=st.text_input(
                    q.get("question",qid)+( " *" if q.get("required") else ""),
                    value=st.session_state.s72_answers.get(qid,""),
                    key=f"s72_answer_{qid}",help=q.get("reason"))
            if st.button("Re-evaluate with answers"):
                try:
                    revised=reconcile(r,dict(st.session_state.s72_answers))
                    st.session_state.s72_replanned=revised
                    st.session_state.s72_ai=None
                    invalidate()
                    st.rerun()
                except Exception as exc: st.error(f"Re-evaluation failed: {exc}")
        if st.session_state.s72_replanned:
            revised=st.session_state.s72_replanned
            st.info("Re-evaluated Stage 5 plan available. Catalogue-verified answers only.")
            with st.expander("Re-evaluation details",expanded=True): st.json(revised)
            if st.button("Apply re-evaluated plan to editor"):
                new=recommendations(revised,funcs)
                st.session_state.s72_steps=[
                    {"id":i+1,"api_key":x["api_key"],"reason":x["reason"],
                     "parameters":default_params(funcs[x["api_key"]],revised)}
                    for i,x in enumerate(new) if x["api_key"] in funcs]
                st.session_state.s72_nextid=len(st.session_state.s72_steps)+1
                st.session_state.s72_result=revised
                st.session_state.s72_replanned=None
                clear_editor_widgets()
                invalidate()
                st.rerun()
        if st.button("✨ Ask NVIDIA NIM for an ordered API workflow",use_container_width=True):
            try:
                with st.spinner("Planning using catalogue evidence..."):
                    ai=plan_workflow(r,funcs,signals)
                st.session_state.s72_ai=ai
                say("assistant","I've prepared an AI API-sequence proposal. Review it before applying; nothing is approved automatically.")
                st.rerun()
            except Exception as exc: st.error(f"AI workflow planning failed: {exc}")
        ai=st.session_state.s72_ai
        if ai is not None:
            with st.expander("AI proposal and warnings",expanded=True): st.json(ai)
            seq=ai.get("sequence",[]) if isinstance(ai,dict) else []
            if seq and st.button("Apply AI sequence to editor"):
                new=[]
                for i,item in enumerate(seq):
                    key=item.get("api_key")
                    if key not in funcs: st.error(f"Unknown API {key!r}; cannot apply."); st.stop()
                    params=default_params(funcs[key],r)
                    params.update({k:str(v) for k,v in (item.get("parameters") or {}).items() if k in params})
                    new.append({"id":i+1,"api_key":key,"reason":item.get("reason","AI proposed"),"parameters":params})
                clear_editor_widgets()
                st.session_state.s72_steps=new
                st.session_state.s72_nextid=len(new)+1
                invalidate(); st.rerun()

    with st.container(border=True):
        st.subheader("2 · Review and edit API workflow")
        steps=st.session_state.s72_steps
        for i,step in enumerate(list(steps)):
            with st.expander(f"{i+1}. {step['api_key']} — {step.get('reason','')}",expanded=True):
                c1,c2,c3=st.columns([1,1,1])
                if c1.button("↑ Move",key=f"s72_up_{step['id']}",disabled=i==0):
                    steps[i-1],steps[i]=steps[i],steps[i-1];invalidate();st.rerun()
                if c2.button("↓ Move",key=f"s72_down_{step['id']}",disabled=i==len(steps)-1):
                    steps[i+1],steps[i]=steps[i],steps[i+1];invalidate();st.rerun()
                if c3.button("Remove",key=f"s72_rm_{step['id']}"):
                    steps.pop(i);invalidate();st.rerun()
                freq=api_frequency()
                recommended=[x["api_key"] for x in recommendations(r,funcs)]
                options=list(dict.fromkeys(recommended +
                    [x for x,_ in sorted(freq.items(),key=lambda pair:-pair[1]) if x in funcs] + sorted(funcs)))
                def api_label(k):
                    return k+(" · Recommended" if k in recommended else
                              f" · Used in {freq[k]} SDs" if freq.get(k) else " · Catalogue")
                key=st.selectbox("API",options,index=options.index(step["api_key"]),format_func=api_label,key=f"s72_api_{step['id']}")
                if key!=step["api_key"]:
                    step.update(api_key=key,parameters=default_params(funcs[key],r),reason="Engineer selected")
                    invalidate();st.rerun()
                schema=funcs[key].get("parameters") or {}
                if not schema: st.error("API parameter schema missing; validation will block approval.")
                for field,meta in schema.items():
                    step["parameters"][field]=st.text_input(
                        field,value=str(step["parameters"].get(field,"")),
                        key=f"s72_p_{step['id']}_{field}",
                        help=f"Type: {meta.get('type','—')} · Direction: {meta.get('direction','—')}")
        if st.button("＋ Add API step"):
            key=sorted(funcs)[0]
            steps.append({"id":st.session_state.s72_nextid,"api_key":key,
                          "reason":"Engineer added","parameters":default_params(funcs[key],r)})
            st.session_state.s72_nextid+=1
            invalidate();st.rerun()

        missing=unknown_signals(steps,funcs,signals)
        if missing:
            st.warning("Unknown signal(s) require verified metadata before validation.")
            for sig in missing:
                with st.expander(f"Register signal: {sig}",expanded=True):
                    tams=st.text_input("Signal-specific TAMS ID",key=f"s72_sig_tams_{sig}")
                    path=st.text_input("Model path",key=f"s72_sig_path_{sig}")
                    kind=st.selectbox("Signal type",TYPES,key=f"s72_sig_type_{sig}")
                    desc=st.text_input("Description (optional)",key=f"s72_sig_desc_{sig}")
                    confirm=st.checkbox("I verified this metadata for the target bench",key=f"s72_sig_ok_{sig}")
                    if st.button("Register in local overlay",key=f"s72_sig_reg_{sig}",
                                 disabled=not(tams.strip() and path.strip() and confirm)):
                        try:
                            register_signal({"signal_key":sig,"tams_id":tams,"model_path":path,
                                             "type":kind,"description":desc},dict(base_signals))
                            invalidate();st.rerun()
                        except Exception as exc: st.error(str(exc))

    with st.container(border=True):
        st.subheader("3 · Parameter bindings")
        st.caption("Choose how each Gherkin parameter becomes an AutomationDesk value. Mappings are engineer-provided and must be approved.")
        reuse=st.session_state.s72_reuse
        placeholder_source={"query":r.get("query"),"original_stage5_plan":({**(r.get("plan") or {}),"proposed_keyword_pattern":reuse['pattern']} if reuse else r.get("plan"))}
        names=keyword_placeholders(placeholder_source)
        if not names:
            st.info("No Gherkin placeholders found. Quote a value or use a Stage 5 pattern containing {status}.")
        for name in names:
            cfg=st.session_state.s72_bindings.get(name,{"mode":"direct","mapping":{}})
            with st.expander(f"Parameter: {{{name}}}",expanded=True):
                options={"Direct value":"direct","Value mapping":"value_map","Signal mapping":"signal_map"}
                current=cfg.get("mode","direct")
                choice=st.selectbox("Binding strategy",list(options),index=list(options.values()).index(current),key=f"s72_bind_mode_{name}")
                mode=options[choice]
                mapping=cfg.get("mapping",{})
                if mode!='direct':
                    st.caption("Enter each Gherkin input and its corresponding approved API value. No special syntax needed.")
                    rows_key=f"s72_map_rows_{name}"
                    if rows_key not in st.session_state:
                        st.session_state[rows_key]=[
                            {"id":i,"input":str(k),"output":str(v)}
                            for i,(k,v) in enumerate(mapping.items())
                        ] or [{"id":0,"input":"","output":""}]
                    rows=st.session_state[rows_key]
                    col_input,col_output,col_action=st.columns([4,4,1])
                    col_input.markdown("**Gherkin input**")
                    col_output.markdown("**API value**")
                    col_action.markdown("**Remove**")
                    for row in list(rows):
                        row_id=row["id"]
                        a,b,c=st.columns([4,4,1])
                        row["input"]=a.text_input(
                            "Gherkin input",value=row["input"],
                            key=f"s72_map_in_{name}_{row_id}",label_visibility="collapsed",
                            placeholder="e.g. open")
                        row["output"]=b.text_input(
                            "API value",value=row["output"],
                            key=f"s72_map_out_{name}_{row_id}",label_visibility="collapsed",
                            placeholder="Approved value")
                        if c.button("✕",key=f"s72_map_remove_{name}_{row_id}",
                                    disabled=len(rows)==1,help="Remove mapping"):
                            st.session_state[rows_key]=[r for r in rows if r["id"]!=row_id]
                            invalidate()
                            st.rerun()
                    if st.button("＋ Add mapping",key=f"s72_map_add_{name}"):
                        next_id=max(r["id"] for r in rows)+1
                        rows.append({"id":next_id,"input":"","output":""})
                        invalidate()
                        st.rerun()
                    parsed={}
                    duplicates=[]
                    incomplete=False
                    for row in rows:
                        k=row["input"].strip()
                        v=row["output"].strip()
                        if not k and not v:
                            continue
                        if not k or not v:
                            incomplete=True
                            continue
                        if k in parsed:
                            duplicates.append(k)
                        else:
                            parsed[k]=v
                    if duplicates:
                        st.error("Duplicate Gherkin inputs: "+", ".join(sorted(set(duplicates))))
                    if incomplete:
                        st.warning("Complete both fields for each mapping row, or remove the incomplete row.")
                    if parsed:
                        st.caption("Mapping preview: "+", ".join(f"{k} → {v}" for k,v in parsed.items()))
                    mapping=parsed
                    st.session_state[f"s72_map_error_{name}"]=bool(duplicates or incomplete)
                else:
                    mapping={}
                    st.session_state.pop(f"s72_map_error_{name}",None)
                st.session_state.s72_bindings[name]={"mode":mode,"mapping":mapping}
        st.caption("For mapped values, set the corresponding API argument to {status} (or your placeholder). Signal fields may use {door} only with Signal mapping.")

    with st.container(border=True):
        st.subheader("4 · Required metadata")
        st.multiselect("Gherkin decorators",["given","when","then"],key="s72_types",
                       help="Engineer selects one or more step types. Never inferred.")
        c1,c2=st.columns(2)
        c1.text_input("Platform decorator",key="s72_platform",placeholder="TAMS-123")
        c2.text_input("TAMS decorator IDs",key="s72_tams",placeholder="123")
        sig=current_signature(steps,signals)
        if st.session_state.s72_signature!=sig and (st.session_state.s72_validation or st.session_state.s72_approved):
            invalidate()
            st.warning("Workflow changed; previous validation and approval were invalidated.")
        if st.button("Validate workflow",type="primary",use_container_width=True):
            validation_steps=copy.deepcopy(steps)
            for step in validation_steps:
                for field,value in step['parameters'].items():
                    match=__import__('re').fullmatch(r'\{([A-Za-z_][A-Za-z0-9_]*)\}',str(value))
                    if match:
                        cfg=st.session_state.s72_bindings.get(match.group(1),{})
                        if cfg.get('mode')=='signal_map' and cfg.get('mapping'):
                            step['parameters'][field]=next(iter(cfg['mapping'].values()))
            result=validate_workflow(r,validation_steps,funcs,signals,
                       {"platform":st.session_state.s72_platform,"tams":st.session_state.s72_tams})
            for binding_name in names:
                if st.session_state.get(f"s72_map_error_{binding_name}",False):
                    result["valid"]=False
                    result.setdefault("errors",[]).append(
                        f"Complete or correct the mapping rows for {{{binding_name}}}.")
            try:
                _check_bindings(st.session_state.s72_bindings,names,signals)
            except GenerationError as exc:
                result["valid"]=False
                result.setdefault("errors",[]).append(str(exc))
            if not st.session_state.s72_types:
                result["valid"]=False
                result.setdefault("errors",[]).append("Select at least one Gherkin step type.")
            if result.get('valid'):
                result['api_sequence']=[{**x,'parameters':copy.deepcopy(steps[i]['parameters'])} for i,x in enumerate(result['api_sequence'])]
            result["metadata"]["step_types"]=list(st.session_state.s72_types)
            result["parameter_bindings"]=copy.deepcopy(st.session_state.s72_bindings)
            st.session_state.s72_validation=result
            st.session_state.s72_signature=sig
            st.session_state.s72_approved=None
            st.session_state.s72_generated=None
            st.rerun()
        validation=st.session_state.s72_validation
        if validation:
            if validation.get("valid"): st.success("Validated. Review and approve explicitly.")
            else:
                for err in validation.get("errors",[]): st.error(err)
        if st.button("Approve validated workflow",disabled=not(validation and validation.get("valid") and st.session_state.s72_signature==sig),
                     use_container_width=True):
            st.session_state.s72_approved=approved_workflow(r,validation)
            st.session_state.s72_approved["parameter_bindings"]=copy.deepcopy(validation.get("parameter_bindings",{}))
            st.session_state.s72_generated=None
            say("assistant","Your validated workflow is approved. You can now generate and download the Python step definition.")
            st.rerun()
        if st.button("Reject workflow",use_container_width=True):
            invalidate()
            say("assistant","Workflow rejected. Review or edit the API sequence and validate again.")
            st.rerun()
        if st.session_state.s72_approved:
            st.success("Engineer-approved workflow")
            st.download_button("Download approved workflow JSON",
                json.dumps(st.session_state.s72_approved,indent=2),
                file_name="approved_api_workflow.json",mime="application/json")

    if st.session_state.s72_approved:
        with st.container(border=True):
            st.subheader("5 · Generate Python SD")
            if st.button("Generate Python preview",type="primary"):
                try:
                    if st.session_state.s72_signature!=current_signature(steps,signals):
                        raise GenerationError("Workflow changed after approval. Validate and approve again.")
                    _check_bindings(st.session_state.s72_approved.get('parameter_bindings',{}),names,signals)
                    ext=(st.session_state.s72_result.get("stage41_evidence") or {}).get("existing_pattern_extension")
                    if ext and ext.get("requested_step_type")=="then":
                        steps=st.session_state.s72_steps
                        setter_keys={"set_signal","set_enum_signal","toggle_signal","set_multiple_signals","ramp_signal"}
                        if not any(x.get("api_key") in ("check_signal","compare_signal") for x in steps) or any(x.get("api_key") in setter_keys for x in steps):
                            raise GenerationError("Then verification requires an approved check/compare API and cannot include signal-setting APIs.")
                    generated=generate(st.session_state.s72_approved,funcs,signals)
                    st.session_state.s72_generated=generated
                    st.session_state.s72_merged=None
                    st.session_state.s72_merge_error=None
                    if ext:
                        original=st.session_state.get("s72_original_source")
                        filename=st.session_state.get("s72_original_filename")
                        if original and filename==ext['sd_file']:
                            try:
                                st.session_state.s72_merged=merge_existing_sd(
                                    original,generated['code'],requested_type=ext['requested_step_type'])
                            except MergeError as exc:
                                st.session_state.s72_merge_error=str(exc)
                        else:
                            st.session_state.s72_merge_error=(
                                "Upload the complete original " + ext['sd_file'] +
                                " in the Analysis section to create a safe merged file. "
                                "RAG code excerpts are not used for merging.")
                    st.rerun()
                except Exception as exc: st.error(str(exc))
            generated=st.session_state.s72_generated
            if generated:
                st.code(generated["code"],language="python")
                st.download_button("Download Python step definition",generated["code"],
                    file_name="generated_step_definition.py",mime="text/x-python")
                merged=st.session_state.s72_merged
                if st.session_state.get("s72_merge_error"):
                    st.warning(st.session_state.s72_merge_error)
                if merged:
                    st.success("Merged preview: existing implementation preserved; new Then function appended.")
                    st.write("**Combined platforms:**",merged["merged_platforms"])
                    st.write("**Combined TAMS IDs:**",", ".join(map(str,merged["merged_tams"])))
                    st.warning("Confirm the runtime accepts comma-separated @platform strings and that APIs/signals work on every added TAMS bench before replacing the original file.")
                    with st.expander("Review added function",expanded=True):
                        st.code(merged['new_function'],language="python")
                    with st.expander("Review complete merged SD file"):
                        st.code(merged['code'],language="python")
                    st.download_button("Download UPDATED existing SD file",merged['code'],
                        file_name=(st.session_state.s72_result.get('stage41_evidence') or {}).get('existing_pattern_extension',{}).get('sd_file','updated_step_definition.py'),
                        mime="text/x-python")
                    st.caption("Preview/download only. Original project file is never overwritten automatically. Review and replace it through your normal approval process.")
                for warning in generated.get("warnings",[]): st.caption(warning)
