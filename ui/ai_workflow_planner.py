"""Stage 5.6: bounded NVIDIA NIM workflow suggestions, catalogue constrained."""
import json
import os
from urllib import request, error

SYSTEM = '''You are an engineering workflow planner, not a code generator.
Return ONLY a JSON object: {"operations":[{"api_key":"catalogue key or null","purpose":"short", "parameters":{"Typed.Name":"value"},"reason":"why"}],"questions":["..."],"notes":"..."}.
Decompose the entire Gherkin requirement into operations in execution order.
Use ONLY API keys supplied in allowed_apis; use ONLY parameter names listed for each API.
Do not invent signals, model paths, TAMS IDs, or values. Unknowns must be blank and asked about.
Existing SD evidence is examples, not proof of full API schemas.
No Python code, no markdown, no explanation outside JSON.'''


def plan_workflow(stage5, functions, signals, *, api_key=None, model=None, timeout=90):
    key = api_key or os.environ.get('NVIDIA_API_KEY')
    if not key:
        raise ValueError('NVIDIA_API_KEY is missing. Configure it before AI planning.')
    allowed = {}
    for k, rec in functions.items():
        allowed[k] = {'purpose':rec.get('purpose',''),
                      'block_type':rec.get('block_type',''),
                      'parameters':list((rec.get('parameters') or {}).keys()),
                      'defaults':rec.get('default_values') or {}}
    payload = {'keyword':stage5.get('query'),
               'keyword_status':stage5.get('stage41_status'),
               'stage5_plan':stage5.get('plan'),
               'retrieval_evidence':stage5.get('stage41_evidence'),
               'allowed_apis':allowed,
               'known_signal_keys':sorted(signals)}
    model = model or os.environ.get('NVIDIA_MODEL','google/gemma-3-12b-it')
    body = {'model':model,'messages':[{'role':'system','content':SYSTEM},
            {'role':'user','content':json.dumps(payload,default=str)}],
            'temperature':0.1,'max_tokens':3000,'stream':False}
    req=request.Request('https://integrate.api.nvidia.com/v1/chat/completions',
        data=json.dumps(body).encode('utf-8'),headers={'Authorization':'Bearer '+key,
        'Content-Type':'application/json'},method='POST')
    try:
        with request.urlopen(req,timeout=timeout) as resp:
            response=json.load(resp)
    except error.HTTPError as exc:
        raise RuntimeError('NVIDIA planning request failed (HTTP %s)' % exc.code) from exc
    content=response['choices'][0]['message']['content']
    # Stage 5's JSON parser requires a Stage 5 plan schema, so parse
    # the distinct Stage 5.6 operations schema independently.
    import re
    cleaned=content.strip()
    cleaned=re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    try:
        parsed=json.loads(cleaned)
    except json.JSONDecodeError:
        decoder=json.JSONDecoder()
        parsed=None
        for match in re.finditer(r"\{", cleaned):
            try:
                candidate,_=decoder.raw_decode(cleaned[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate,dict) and isinstance(candidate.get('operations'),list):
                parsed=candidate
                break
        if parsed is None:
            raise ValueError('NVIDIA returned no complete workflow JSON object.')
    if not isinstance(parsed,dict) or not isinstance(parsed.get('operations'),list):
        raise ValueError('AI response does not contain an operations list.')
    errors=[]; sequence=[]
    for i,op in enumerate(parsed['operations'],1):
        if not isinstance(op,dict): errors.append(f'Operation {i} is not an object');continue
        api=op.get('api_key')
        if api not in functions:
            errors.append(f'Operation {i}: unknown or unselected API {api!r}; manual selection required')
            continue
        schema=(functions[api].get('parameters') or {})
        params=op.get('parameters') or {}
        if not isinstance(params,dict):
            errors.append(f'Operation {i}: invalid parameter map');params={}
        unknown=set(params)-set(schema)
        if unknown:
            errors.append(f'Operation {i}: invalid parameters {sorted(unknown)}')
            params={k:v for k,v in params.items() if k in schema}
        complete={name:str(params.get(name,'')) for name in schema}
        # Values may be unknown; the engineer will fill them before approval.
        sequence.append({'api_key':api,'reason':str(op.get('purpose') or op.get('reason') or 'AI suggestion'),
                         'parameters':complete,'source':'nvidia_nim_suggestion'})
    return {'sequence':sequence,'errors':errors,
            'questions':parsed.get('questions') or [],'notes':parsed.get('notes') or '',
            'model':model,'source':'nvidia_nim','requires_human_review':True}
