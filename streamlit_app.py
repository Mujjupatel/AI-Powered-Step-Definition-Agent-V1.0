import os
import json as _json
import streamlit as st
from ui.state import init_state,reset_analysis,set_result,all_required_answered,approved_envelope
from ui.workflow import analyze_keyword,analyze_feature,extract_feature_steps
from ui.components import banner,plan_view,evidence_view,parameter_view,questions_view,extras
from ui.replan import reconcile
from ui.ai_workflow_planner import plan_workflow
from ui.stage6_generator import generate, GenerationError
from ui.signal_registration import combined_signals, unknown_signals, register_signal, TYPES
from ui.workflow_editor import load_catalogue, api_frequency, recommendations, default_params, validate_workflow, approved_workflow

st.set_page_config(page_title="SD Definition Agent",page_icon="🧩",layout="wide")
init_state()
st.title("Step Definition Agent")
st.caption("RAG + NVIDIA NIM + human approval · Stage 6 deterministic SD preview (POC).")

with st.sidebar:
    st.header("Planner")
    mode=st.selectbox("Mode",["nim","offline"])
    st.caption("Model: "+os.getenv("NVIDIA_MODEL","google/gemma-3-12b-it"))
    top_files=st.slider("Similar SD files",1,6,3)
    candidate_k=st.slider("Candidate retrieval depth",1,10,5)
    st.divider(); st.markdown("**Stage 6**")
    st.caption("Generate from an approved workflow below.")

single,feature=st.tabs(["Single Keyword","Feature File"])
with single:
    q=st.text_area("Gherkin keyword",placeholder="the steering angle is set to 30 degrees",height=100)
    a,b=st.columns(2)
    if a.button("Analyze keyword",type="primary",use_container_width=True):
        try:
            with st.spinner("Running Stage 4.1 + Stage 5..."):
                set_result(analyze_keyword(q,mode,top_files,candidate_k))
                st.session_state.pop("replanned_result", None)
                st.session_state.pop("ai_workflow_result", None)
        except Exception as e: st.exception(e)
    if b.button("Clear",use_container_width=True):
        reset_analysis()
        st.session_state.pop("replanned_result", None)
        st.rerun()

with feature:
    up=st.file_uploader("Upload .feature",type=["feature","txt"])
    pasted=st.text_area("Or paste feature",height=200)
    ft=pasted
    if up:
        ft=up.getvalue().decode("utf-8",errors="replace")
    if ft:
        steps=extract_feature_steps(ft); st.caption(f"Detected {len(steps)} executable step(s).")
        with st.expander("Detected steps"):
            for i,s in enumerate(steps,1): st.write(f"{i}. {s}")
    if st.button("Analyze feature",type="primary",use_container_width=True):
        try:
            with st.spinner("Analyzing feature..."):
                st.session_state.feature_results=analyze_feature(ft,mode,top_files,candidate_k)
        except Exception as e: st.exception(e)
    for i,r in enumerate(st.session_state.get("feature_results",[]),1):
        with st.expander(f'{i}. {r.get("query")} · {r.get("stage41_status")}',expanded=i==1):
            banner(r); plan_view(r); extras(r)

r=st.session_state.analysis_result
if r:
    st.divider(); banner(r); plan_view(r); extras(r); evidence_view(r); parameter_view(r); questions_view(r)
    if (r.get("plan") or {}).get("questions"):
        if st.button("Re-evaluate with Answers", type="primary", use_container_width=True):
            try:
                st.session_state.replanned_result = reconcile(r, dict(st.session_state.human_answers))
                st.session_state.decision = None
                st.session_state.approved_plan = None
            except Exception as exc:
                st.exception(exc)

    revised = st.session_state.get("replanned_result")
    active = revised if revised is not None else r
    if revised is not None:
        st.subheader("Re-evaluated plan")
        v = revised.get("validation") or {}
        if v.get("valid") and (revised.get("plan") or {}).get("ready_for_approval"):
            st.success("Re-evaluation passed. Plan is ready for explicit approval.")
        else:
            st.warning("Re-evaluation is blocked; review the validation errors below.")
            for err in v.get("errors") or []:
                st.error(str(err))
        with st.expander("Re-evaluation details"):
            st.json(revised)


    st.subheader("API workflow editor")
    st.caption("AI proposes a sequence. You may replace, add, remove, or reorder any catalogue API. Every step has its own parameter form.")
    try:
        functions, signal_catalogue = load_catalogue()
    except Exception as exc:
        st.error(f"Could not load full catalogue: {exc}")
        functions, signal_catalogue = {}, {}
    if functions:
        source_signals = dict(signal_catalogue)
        try:
            signal_catalogue = combined_signals(source_signals)
        except (ValueError, OSError) as exc:
            st.error(f"Signal overlay could not be loaded: {exc}")
            st.stop()
        initial = recommendations(r, functions)
        fingerprint = (r.get("query"), id(r))
        if st.session_state.get("workflow_fingerprint") != fingerprint:
            st.session_state.workflow_fingerprint = fingerprint
            st.session_state.workflow_steps = [
                {"id":i+1, "api_key":x["api_key"],
                 "reason":x["reason"],"parameters":default_params(functions[x["api_key"]],r)}
                for i,x in enumerate(initial)
            ]
            st.session_state.workflow_validation = None
            st.session_state.workflow_approved = None
        st.markdown("**AI multi-API workflow suggestion**")
        st.caption("NVIDIA NIM proposes an ordered sequence; no step is approved automatically.")
        if st.button("Generate AI API sequence", key="generate_ai_workflow", use_container_width=True):
            try:
                with st.spinner("Planning ordered operations using NVIDIA NIM and catalogue evidence..."):
                    ai = plan_workflow(r, functions, signal_catalogue)
                st.session_state.ai_workflow_result = ai
            except Exception as exc:
                st.error(f"AI workflow planning failed: {exc}")
        ai_result = st.session_state.get("ai_workflow_result")
        if ai_result:
            st.json({"proposed_steps":ai_result["sequence"],"questions":ai_result["questions"],
                     "warnings":ai_result["errors"],"model":ai_result["model"]})
            st.warning("Review all AI suggestions. Unverified APIs and parameters are not imported.")
            if st.button("Use AI sequence in editor",key="use_ai_workflow",disabled=not ai_result["sequence"]):
                st.session_state.workflow_steps = [
                    {"id":i+1,"api_key":x["api_key"],"reason":x["reason"],
                     "parameters":x["parameters"]} for i,x in enumerate(ai_result["sequence"])]
                st.session_state.workflow_validation = None
                st.session_state.workflow_approved = None
                st.session_state.workflow_validated_signature = None
                # Reset stale parameter widget keys after replacing the sequence.
                for widget_key in list(st.session_state.keys()):
                    if widget_key.startswith(("wf_param_", "wf_api_")):
                        del st.session_state[widget_key]
                st.rerun()
        freq = api_frequency()
        recommended_keys = [x["api_key"] for x in initial]
        popular_keys = [x for x,_ in sorted(freq.items(), key=lambda kv:-kv[1]) if x in functions]
        options = list(dict.fromkeys(recommended_keys + popular_keys + sorted(functions)))
        def api_label(k):
            label = "Recommended" if k in recommended_keys else f"Used in {freq[k]} SDs" if freq.get(k) else "Catalogue"
            return f"{k}  ·  {label}"
        steps = st.session_state.workflow_steps
        for i, step in enumerate(list(steps)):
            with st.container(border=True):
                head, up, down, delete = st.columns([7,1,1,1])
                head.markdown(f"**Step {i+1}** · {step.get('reason','Engineer selected')}")
                if up.button("↑",key=f"wf_up_{step['id']}",disabled=i==0):
                    steps[i-1],steps[i]=steps[i],steps[i-1]
                    st.session_state.workflow_validation=None
                    st.rerun()
                if down.button("↓",key=f"wf_down_{step['id']}",disabled=i==len(steps)-1):
                    steps[i+1],steps[i]=steps[i],steps[i+1]
                    st.session_state.workflow_validation=None
                    st.rerun()
                if delete.button("✕",key=f"wf_del_{step['id']}"):
                    steps.pop(i)
                    st.session_state.workflow_validation=None
                    st.rerun()
                key=st.selectbox("Implementation API",options,index=options.index(step["api_key"]),
                                 format_func=api_label,key=f"wf_api_{step['id']}")
                if key != step["api_key"]:
                    step["api_key"]=key
                    step["parameters"]=default_params(functions[key],r)
                    step["reason"]="Engineer changed API"
                    st.session_state.workflow_validation=None
                    st.rerun()
                api=functions[key]
                st.caption(f"Block type: {api.get('block_type','—')} · {api.get('purpose','')}")
                schema=api.get("parameters") or {}
                if not schema:
                    st.warning("Parameter schema unavailable. Cannot approve this API until its schema is provided.")
                for full, detail in schema.items():
                    val=step["parameters"].get(full,"")
                    step["parameters"][full]=st.text_input(
                        full,value=val,key=f"wf_param_{step['id']}_{full}",
                        help=f"Type: {detail.get('type','—')} · Direction: {detail.get('direction','—')}")
        if st.button("＋ Add API step",use_container_width=True):
            key=recommended_keys[0] if recommended_keys else sorted(functions)[0]
            next_id=max([x["id"] for x in steps],default=0)+1
            steps.append({"id":next_id,"api_key":key,"reason":"Engineer added step",
                          "parameters":default_params(functions[key],r)})
            st.session_state.workflow_validation=None
            st.rerun()
        missing_signals=unknown_signals(steps,functions,signal_catalogue)
        if missing_signals:
            st.subheader("Register missing signals")
            st.info("Unknown signals must be registered with signal-specific TAMS ID, model path and type. The original catalogue will not be modified.")
            for missing_key in missing_signals:
                with st.expander(f"Register {missing_key}", expanded=True):
                    # Standalone widgets update session state on each interaction,
                    # unlike st.form inputs which are submitted only as a batch.
                    st.text_input("Signal key", value=missing_key, disabled=True,
                                  key=f"sig_key_display_{missing_key}")
                    sig_tams=st.text_input("Signal TAMS ID", key=f"sig_tams_{missing_key}",
                        help="Signal-specific TAMS ID; separate from SD decorator metadata.")
                    sig_path=st.text_input("Model path", key=f"sig_path_{missing_key}")
                    sig_type=st.selectbox("Signal type", TYPES, key=f"sig_type_{missing_key}")
                    sig_description=st.text_input("Description (optional)", key=f"sig_desc_{missing_key}")
                    acknowledged=st.checkbox(
                        "I verified this signal's engineering metadata and authorize local registration.",
                        key=f"sig_confirm_{missing_key}")
                    entry={"signal_key":missing_key,"tams_id":sig_tams,
                           "model_path":sig_path,"type":sig_type,"description":sig_description}
                    st.json(entry)
                    fields_ready=bool(sig_tams.strip() and sig_path.strip() and sig_type)
                    if not fields_ready:
                        st.caption("Signal TAMS ID and model path are required to register this signal.")
                    submitted=st.button("Register signal in local overlay",
                        key=f"sig_register_button_{missing_key}",
                        disabled=not (fields_ready and acknowledged))
                    if submitted:
                        try:
                            register_signal(entry,source_signals)
                        except (ValueError,OSError) as exc:
                            st.error(str(exc))
                        else:
                            st.session_state.workflow_validation=None
                            st.session_state.workflow_approved=None
                            st.session_state.workflow_validated_signature=None
                            st.success(f"Registered {missing_key} in local approved signal overlay.")
                            st.rerun()
        st.markdown("**Gherkin step types (engineer-selected)**")
        selected_step_types = st.multiselect(
            "Select one or more decorators",
            ["given", "when", "then"],
            default=[],
            key="wf_step_types",
            format_func=lambda value: value.title(),
            help="Required. Choose exactly which Behave decorators to generate."
        )
        if not selected_step_types:
            st.warning("Select at least one Gherkin step type before validation.")
        st.markdown("**Step-definition metadata**")
        meta1,meta2=st.columns(2)
        platform=meta1.text_input("Platform decorator value",key="wf_platform",placeholder="TAMS-123")
        tams=meta2.text_input("TAMS decorator value",key="wf_tams",placeholder="123")
        st.caption("The API sequence and metadata will be saved as a structured, human-approved plan. No Python is generated.")
        import json as _json
        if st.button("Validate complete workflow",type="primary",use_container_width=True):
            st.session_state.workflow_validation=validate_workflow(
                r,steps,functions,signal_catalogue,{"platform":platform,"tams":tams})
            if not selected_step_types:
                st.session_state.workflow_validation = {"valid": False, "errors": ["Select at least one Gherkin step type."], "api_sequence": [], "metadata": {}}
            else:
                st.session_state.workflow_validation["metadata"]["step_types"] = list(selected_step_types)
            st.session_state.workflow_approved=None
            st.session_state.workflow_validated_signature=_json.dumps({"steps":steps,"platform":platform,"tams":tams,"step_types":selected_step_types,"signals":signal_catalogue},sort_keys=True)
        # Invalidate an earlier successful validation whenever any input changes.
        import json as _json
        current_signature=_json.dumps({"steps":steps,"platform":platform,"tams":tams,"step_types":selected_step_types,"signals":signal_catalogue},sort_keys=True)
        if st.session_state.get("workflow_validated_signature") != current_signature:
            st.session_state.workflow_validation=None
            st.session_state.workflow_approved=None
            st.session_state.stage6_output=None
        validation=st.session_state.get("workflow_validation")
        if validation:
            if validation["valid"]:
                st.success("Complete workflow validated. Review and explicitly approve.")
                with st.expander("Validated ordered API plan",expanded=True):
                    st.json({"api_sequence":validation["api_sequence"],"metadata":validation["metadata"]})
            else:
                st.error("Workflow validation blocked approval.")
                for error in validation["errors"]: st.write("• "+error)
        c1,c2=st.columns(2)
        if c1.button("Approve complete workflow",disabled=not(validation and validation["valid"]),use_container_width=True):
            st.session_state.workflow_approved=approved_workflow(r,validation)
        if c2.button("Reject workflow",use_container_width=True):
            st.session_state.workflow_approved=None
            st.session_state.workflow_validation=None
            st.warning("Workflow rejected.")
        if st.session_state.get("workflow_approved"):
            st.success("Workflow approved. Stage 6 preview is available below.")
            st.download_button("Download approved workflow JSON",
                data=__import__("json").dumps(st.session_state.workflow_approved,indent=2),
                file_name="approved_api_workflow.json",mime="application/json")

        # Stage 6: code generation is independent of the LLM and uses exact schemas.
        if st.session_state.get("workflow_approved"):
            st.divider()
            st.subheader("Stage 6 · Generate Behave SD (POC)")
            st.caption("Deterministic generation from the approved workflow and exact Function Catalogue schemas. No execution is performed.")
            if st.button("Generate Python SD preview", key="stage6_generate", use_container_width=True):
                try:
                    # Revalidate at generation time, not just at approval time.
                    fresh = validate_workflow(r, steps, functions, signal_catalogue,
                                              {"platform": platform, "tams": tams})
                    if selected_step_types:
                        fresh["metadata"]["step_types"] = list(selected_step_types)
                    if not fresh["valid"] or not selected_step_types:
                        raise GenerationError("Workflow changed or failed validation: " + "; ".join(fresh["errors"]))
                    from ui.workflow_editor import approved_workflow as _approved_workflow
                    current_approved = _approved_workflow(r, fresh)
                    if current_approved["api_sequence"] != st.session_state.workflow_approved["api_sequence"] or current_approved["metadata"] != st.session_state.workflow_approved["metadata"]:
                        raise GenerationError("Approved workflow is stale. Validate and approve again.")
                    st.session_state.stage6_output = generate(current_approved, functions, signal_catalogue)
                except (GenerationError, ValueError, SyntaxError) as exc:
                    st.session_state.stage6_output = None
                    st.error(str(exc))
            generated = st.session_state.get("stage6_output")
            if generated:
                st.code(generated["code"], language="python")
                st.caption(f"{generated['api_count']} API calls · Signals: {', '.join(generated['signals_used']) or 'none'}")
                for warning in generated["warnings"]:
                    st.warning(warning)
                st.download_button("Download generated step definition (.py)",
                                   data=generated["code"],
                                   file_name="generated_step_definition.py",
                                   mime="text/x-python", key="stage6_download")
