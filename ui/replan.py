"""Deterministic Stage 5.5 human-evidence reconciliation. Never invent catalogue facts."""
import copy
import re


def _tokens(value):
    return [x.strip() for x in re.split(r'[,;\n]+', str(value)) if x.strip()]


def reconcile(result, answers):
    updated = copy.deepcopy(result)
    plan = updated['plan']
    evidence = updated.get('stage41_evidence') or {}
    questions = plan.get('questions') or []
    normalized = {str(k): str(v).strip() for k, v in answers.items() if str(v).strip()}
    known_signals = {s.get('signal_key') for s in evidence.get('known_signals', [])}
    known_apis = {a.get('api_key') for a in evidence.get('confirmed_apis', [])}
    candidate_apis = {a.get('api_key') for a in evidence.get('candidate_apis', [])}
    allowed_apis = known_apis | candidate_apis
    errors = list((updated.get('validation') or {}).get('errors') or [])
    unresolved = []
    confirmations = []

    for q in questions:
        qid = str(q.get('id', ''))
        if not q.get('required'):
            continue
        answer = normalized.get(qid)
        if not answer:
            unresolved.append(q)
            continue
        qtext = (qid + ' ' + str(q.get('question', ''))).lower()
        if 'signal' in qtext and ('catalogue' in qtext or 'signal' in qid.lower()):
            supplied = _tokens(answer)
            unknown = [x for x in supplied if x not in known_signals]
            if unknown:
                errors.append('Signal(s) not verified in current Signal Catalogue: ' + ', '.join(unknown))
                unresolved.append(q)
                continue
            plan['selected_signals'] = list(dict.fromkeys((plan.get('selected_signals') or []) + supplied))
            confirmations.append({'kind': 'signal', 'value': supplied, 'source': 'catalogue_verified_human_selection'})
        elif ('api' in qtext or 'function' in qtext):
            supplied = _tokens(answer)
            unknown = [x for x in supplied if x not in allowed_apis]
            if unknown:
                errors.append('API(s) not verified in current retrieved Function Catalogue evidence: ' + ', '.join(unknown))
                unresolved.append(q)
                continue
            proposed = {x.get('api_key'): x for x in plan.get('proposed_apis', [])}
            for key in supplied:
                if key not in proposed:
                    proposed[key] = {'api_key': key, 'reason': 'Explicit human selection from retrieved catalogue'}
                proposed[key]['status'] = 'confirmed' if key in known_apis else 'human_approved_candidate'
            plan['proposed_apis'] = list(proposed.values())
            confirmations.append({'kind': 'api', 'value': supplied, 'source': 'human_selected_catalogue_candidate'})
        elif 'platform' in qtext or 'tams' in qtext:
            # Decorator metadata is separate from API parameters.
            if not re.search(r'\bTAMS\b|@platform|@TAMS|platform\s*[:=]', answer, re.I):
                errors.append('Specify platform/TAMS decorator metadata explicitly (e.g. @platform("TAMS-123"), @TAMS([123])).')
                unresolved.append(q)
                continue
            confirmations.append({'kind': 'decorator_metadata', 'value': answer, 'source': 'human_supplied_unverified'})
        else:
            confirmations.append({'kind': 'additional_answer', 'question_id': qid, 'value': answer, 'source': 'human_supplied_unverified'})

    # Never claim a plan complete without an authoritative signal and an approved catalogue API.
    if not plan.get('selected_signals'):
        errors.append('No catalogue-verified signal selected.')
    if not plan.get('proposed_apis'):
        errors.append('No catalogue-verified API selected.')
    for item in plan.get('proposed_apis', []):
        if item.get('status') == 'candidate_requires_approval':
            errors.append('Candidate API still needs explicit confirmation: ' + str(item.get('api_key')))
    plan['questions'] = unresolved
    plan['ready_for_generation'] = False
    ready = not errors and not unresolved
    plan['ready_for_approval'] = ready
    updated['validation'] = {'valid': not errors, 'errors': list(dict.fromkeys(errors)), 'closed_list_enforced': True}
    updated['approval'] = {'status': 'PENDING_APPROVAL' if ready else 'PENDING_USER_INPUT' if unresolved else 'BLOCKED_BY_VALIDATION', 'required_before_stage6': True}
    updated['human_evidence'] = {'answers': normalized, 'confirmations': confirmations, 'unresolved_question_ids': [str(q.get('id')) for q in unresolved]}
    updated['replanned'] = True
    return updated
