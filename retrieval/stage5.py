import json
import os
import socket
import time

from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib import request, error

from .stage41 import Stage41Retriever


# ============================================================
# NVIDIA NIM SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the planning component of a dSPACE AutomationDesk
Behave step-definition generation agent.

Your job is to create a GENERATION PLAN ONLY.

NEVER generate Python source code.

Stage 4.1 provides retrieval evidence, catalogue facts,
existing implementation evidence, candidate APIs, confirmed
APIs, and missing information.


============================================================
CLOSED-LIST AUTHORITY RULES
============================================================

Treat Stage 4.1 catalogue facts as closed-list authority.

Never invent:

- API keys
- signal keys
- TAMS IDs
- model paths
- API parameter names
- API parameter types

You may only use signals supplied in:

    stage41.known_signals

You may only use APIs supplied in:

    stage41.confirmed_apis
    stage41.candidate_apis


============================================================
API APPROVAL RULES
============================================================

For EXACT_EXISTING:

- Preserve the observed implementation pattern.
- APIs supplied as confirmed_apis may be treated as confirmed.
- Existing parameter mappings may be reused when provided.

For NEW_WITH_KNOWN_CATALOGUE:

- candidate_apis are proposals only.
- Candidate APIs MUST remain:
      candidate_requires_approval
- Never promote a candidate API to confirmed.
- Candidate APIs require explicit user approval before
  Stage 6 generation.

IMPORTANT:

Knowing the schema of a candidate API does NOT mean that the
candidate API itself has been approved.

You MAY use the known parameter schema of a candidate API
for planning.

Example:

Candidate API:
    set_signal

Known parameters:
    String.Variable
    Variant.value
    Float.PostDelay

You may create a parameter plan using those parameters.

However:

    set_signal

must still remain:

    candidate_requires_approval


============================================================
INSUFFICIENT KNOWLEDGE
============================================================

For INSUFFICIENT_KNOWLEDGE:

- Do not guess.
- Do not invent signals.
- Do not invent APIs.
- Do not invent catalogue information.
- Ask the user for the missing information.


============================================================
STEP-DEFINITION METADATA RULES
============================================================

@platform(...) and @TAMS(...) are step-definition
decorators/metadata.

They are NOT Function Catalogue API parameters.

Never describe @platform or @TAMS values as API inputs.

If platform/TAMS decorator information is unavailable,
request it separately as step-definition metadata.


============================================================
PARAMETER RULES
============================================================

Only use API parameters supplied by Stage 4.1.

A parameter may receive its value from:

1. the user's keyword/query
2. a confirmed Signal Catalogue reference
3. an existing exact SD implementation
4. a supplied catalogue/default value
5. user confirmation

If a value is explicitly present in the user's keyword,
you may map that value to a compatible known API parameter.

For a signal-setting operation:

- String.Variable should normally use the selected confirmed
  Signal Catalogue signal key/reference.
- A value explicitly present in the keyword may be mapped to
  the compatible value parameter.
- Example:
      query = "the steering angle is set to 30 degrees"
      selected signal = steering_angle

  A valid proposed mapping may be:
      String.Variable = steering_angle
      Variant.value = 30

Do not invent parameter names or types.


============================================================
DEFAULT VALUE RULES
============================================================

If a confirmed or candidate API contains default_values:

- use the provided value as the proposed default
- identify it as supplied/default evidence
- do not invent a different value
- do not replace a known default with TBD unless there is
  genuine ambiguity

Example:

default_values:
    PostDelay: "0.5"

Preferred parameter plan:

    Float.PostDelay = "0.5"

Do NOT unnecessarily ask the user to manually provide a
value that already exists as a supplied default.

The user may still approve or override the proposed value
later.


============================================================
APPROVAL RULES
============================================================

Stage 5 NEVER authorizes final Python generation.

Stage 5 only prepares a plan for user review.

Even for EXACT_EXISTING:

    ready_for_generation MUST be false

If all required information is available:

    ready_for_approval = true
    ready_for_generation = false

If required information is missing:

    ready_for_approval = false
    ready_for_generation = false

Final generation permission happens only after explicit
user approval outside this planning step.


============================================================
OUTPUT FORMAT
============================================================

Return exactly ONE JSON OBJECT.

The top-level response MUST be a JSON object beginning with {
and ending with }.

DO NOT return a JSON array/list as the top-level response.

DO NOT return:

[
    {
        "summary": "..."
    }
]

Return:

{
    "summary": "...",
    "proposed_keyword_pattern": "...",
    "step_type": "when",
    "selected_signals": [],
    "proposed_apis": [],
    "parameter_plan": [],
    "execution_flow": [],
    "questions": [],
    "ready_for_approval": false,
    "ready_for_generation": false
}

Do not wrap the JSON object in explanation text.
Do not use markdown code fences.
Do not generate Python source code.
"""


# ============================================================
# JSON EXTRACTION
# ============================================================

def _extract_json(text: str):
    """
    Extract the complete Stage 5 plan object from an LLM response.

    The NIM model may return:
    - plain JSON
    - markdown fenced JSON
    - reasoning before JSON
    - multiple JSON fragments
    - arrays embedded before the actual plan object

    This function searches for the JSON object that looks like
    a complete Stage 5 plan instead of blindly returning the
    first valid JSON fragment.
    """

    if not text or not text.strip():
        raise ValueError(
            "NVIDIA NIM returned an empty response."
        )

    text = text.strip()

    required_plan_keys = {
        "summary",
        "proposed_keyword_pattern",
        "step_type",
        "selected_signals",
        "proposed_apis",
        "parameter_plan",
        "execution_flow",
        "questions",
        "ready_for_approval",
        "ready_for_generation",
    }

    core_plan_keys = {
        "summary",
        "selected_signals",
        "proposed_apis",
        "parameter_plan",
    }

    def is_plan_object(obj):
        """
        Return True only when obj looks like the Stage 5
        plan dictionary we expect.
        """

        if not isinstance(obj, dict):
            return False

        keys = set(obj.keys())

        # Preferred: complete expected schema.
        if required_plan_keys.issubset(keys):
            return True

        # Fallback for a mostly complete model response.
        if core_plan_keys.issubset(keys):
            return True

        return False

    def find_plan_recursive(obj):
        """
        Search recursively in case the model wraps the plan:

        {
            "plan": {...}
        }

        or:

        [
            {...},
            {"plan": {...}}
        ]
        """

        if is_plan_object(obj):
            return obj

        if isinstance(obj, dict):

            # Prefer obvious wrapper keys first.
            for key in (
                "plan",
                "generation_plan",
                "stage5_plan",
                "result",
                "output",
            ):
                value = obj.get(key)

                if value is not None:
                    found = find_plan_recursive(value)

                    if found is not None:
                        return found

            # Then inspect remaining dictionary values.
            for value in obj.values():
                found = find_plan_recursive(value)

                if found is not None:
                    return found

        elif isinstance(obj, list):

            for value in obj:
                found = find_plan_recursive(value)

                if found is not None:
                    return found

        return None

    # ========================================================
    # 1. Try the complete response directly
    # ========================================================

    try:
        parsed = json.loads(text)

        found = find_plan_recursive(parsed)

        if found is not None:
            return found

    except json.JSONDecodeError:
        pass

    # ========================================================
    # 2. Try markdown fenced content
    # ========================================================

    cleaned = text

    if "```json" in cleaned:
        cleaned = cleaned.split(
            "```json",
            1
        )[1]

        cleaned = cleaned.split(
            "```",
            1
        )[0].strip()

    elif "```" in cleaned:
        cleaned = cleaned.split(
            "```",
            1
        )[1]

        cleaned = cleaned.split(
            "```",
            1
        )[0].strip()

    try:
        parsed = json.loads(cleaned)

        found = find_plan_recursive(parsed)

        if found is not None:
            return found

    except json.JSONDecodeError:
        pass

    # ========================================================
    # 3. Scan ALL possible JSON starting positions
    #
    # IMPORTANT:
    # Do not return the first valid JSON fragment.
    # Search until we find an object that actually looks
    # like the Stage 5 plan.
    # ========================================================

    decoder = json.JSONDecoder()

    discovered_json = []

    for index, char in enumerate(text):

        if char not in ("{", "["):
            continue

        try:
            obj, consumed = decoder.raw_decode(
                text[index:]
            )

            discovered_json.append(obj)

            found = find_plan_recursive(obj)

            if found is not None:
                return found

        except json.JSONDecodeError:
            continue

    # ========================================================
    # 4. No Stage 5 plan found
    # ========================================================

    discovered_types = [
        type(obj).__name__
        for obj in discovered_json[:10]
    ]

    preview = text[:3000]

    raise ValueError(
        "Could not find a complete Stage 5 plan object "
        "in the NVIDIA NIM response.\n\n"
        f"JSON fragments discovered: "
        f"{len(discovered_json)}\n"
        f"Fragment types: {discovered_types}\n\n"
        f"Response preview:\n{preview}"
    )

# ============================================================
# NVIDIA NIM CONFIGURATION
# ============================================================

@dataclass
class NIMConfig:

    api_key: str

    model: str

    base_url: str = (
        "https://integrate.api.nvidia.com/v1"
    )

    timeout: int = 90

    max_tokens: int = 5000

    temperature: float = 0.1


# ============================================================
# NVIDIA NIM CLIENT
# ============================================================

class NIMPlannerClient:

    def __init__(
        self,
        config: NIMConfig
    ):
        self.config = config


    def create_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback only if no meaningful streamed content arrives in 20 seconds,
        or if a model fails. Once content starts, allow it to finish.

        NVIDIA_MODELS: ordered comma-separated IDs.
        NVIDIA_MODEL_TIMEOUT: seconds until first meaningful content (default 20).
        NVIDIA_STREAM_IDLE_TIMEOUT: seconds without more data after content (default 120).
        """
        configured = os.getenv("NVIDIA_MODELS", "").strip()
        models = [m.strip() for m in configured.split(",") if m.strip()] if configured else [self.config.model]
        if not models:
            raise ValueError("No NVIDIA inference models configured")
        first_timeout = max(2.0, float(os.getenv("NVIDIA_MODEL_TIMEOUT", "20")))
        idle_timeout = max(5.0, float(os.getenv("NVIDIA_STREAM_IDLE_TIMEOUT", "120")))
        url = os.getenv("NVIDIA_BASE_URL", self.config.base_url).rstrip("/") + "/chat/completions"
        failures = []
        self.last_model = None
        self.last_latency = None

        for model in models:
            started = time.monotonic()
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, indent=2)},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": True,
            }
            req = request.Request(
                url, data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "Accept": "text/event-stream",
                         "Authorization": "Bearer " + self.config.api_key},
                method="POST",
            )
            try:
                pieces = []
                started_content = False
                done = False
                # urlopen's timeout is an inactivity/socket timeout, not a total deadline.
                with request.urlopen(req, timeout=first_timeout) as response:
                    while True:
                        if not started_content and time.monotonic() - started > first_timeout:
                            raise TimeoutError(f"No meaningful model content within {first_timeout:g}s")
                        line = response.readline()
                        if not line:
                            break
                        line = line.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            done = True
                            break
                        if not data:
                            continue
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("error"):
                            raise ValueError(f"Streaming error: {str(chunk['error'])[:300]}")
                        for choice in chunk.get("choices", []):
                            delta = choice.get("delta") or {}
                            content = delta.get("content") or ""
                            if isinstance(content, str) and content:
                                pieces.append(content)
                                if not started_content:
                                    started_content = True
                                    print(f"[NVIDIA fallback] STREAM START model={model} "
                                          f"after={time.monotonic()-started:.2f}s; waiting for completion")
                                    # Increase socket inactivity allowance only after real content.
                                    # urllib's underlying socket is not part of its public API;
                                    # use the response socket when available.
                                    try:
                                        response.fp.raw._sock.settimeout(idle_timeout)
                                    except (AttributeError, OSError):
                                        pass
                if not started_content:
                    raise ValueError("Stream ended without assistant content")
                if not done:
                    raise ValueError("Stream ended before [DONE]; response may be incomplete")
                plan = _extract_json("".join(pieces))
                if not isinstance(plan, dict):
                    raise ValueError("Expected a Stage 5 JSON plan object")
                self.last_model = model
                self.last_latency = round(time.monotonic() - started, 2)
                print(f"[NVIDIA fallback] SUCCESS model={model} elapsed={self.last_latency}s")
                return plan
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:250]
                if exc.code in (400, 401, 403, 422):
                    raise RuntimeError(f"NVIDIA NIM HTTP {exc.code} on {model}: {detail}") from exc
                reason = f"HTTP {exc.code}: {detail}"
            except (error.URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as exc:
                reason = f"connection/timeout: {type(exc).__name__}: {exc}"
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                reason = f"invalid/incomplete stream: {exc}"
            elapsed = round(time.monotonic() - started, 2)
            failures.append(f"{model} ({elapsed}s): {reason}")
            print(f"[NVIDIA fallback] FAILED model={model} elapsed={elapsed}s: {reason}")
        raise RuntimeError("All NVIDIA model attempts failed:\n" + "\n".join(failures))


# ============================================================
# STAGE 5 PLANNER
# ============================================================


def _safe_gherkin_pattern(query, proposed, status):
    """Preserve fixed Gherkin wording; parameterize only quoted literal values.

    Existing keyword patterns from SDs are preserved when they structurally match.
    New keywords use a conservative template and do not generalize nouns/signals.
    """
    import re
    source = str(query or "").strip()
    source = re.sub(r"^(Given|When|Then|And|But)\\s+", "", source, flags=re.I)
    if status == "EXACT_EXISTING" and proposed:
        pattern = str(proposed).strip()
        regex = re.escape(pattern)
        regex = re.sub(r"\\\{[A-Za-z_][A-Za-z0-9_]*\\\}", r"(.+?)", regex)
        try:
            if re.fullmatch(regex, source, flags=re.I):
                return pattern
        except re.error:
            pass
    # Only explicitly quoted values are variable; keep quotes in Behave pattern.
    matches = list(re.finditer(r"""(["'])(.*?)\1""", source))
    if not matches:
        return source
    chunks = []
    last = 0
    for i, match in enumerate(matches):
        chunks.append(source[last:match.start()])
        literal = match.group(2)
        # Distinguish common value categories without replacing fixed signal names.
        label = "value" if i == 0 else f"value_{i+1}"
        chunks.append(f'{match.group(1)}{{{label}}}{match.group(1)}')
        last = match.end()
    chunks.append(source[last:])
    return "".join(chunks)


class Stage5Planner:

    def __init__(
        self,
        stage41: Stage41Retriever,
        nim_client: Optional[
            NIMPlannerClient
        ] = None,
    ):

        self.stage41 = stage41

        self.nim_client = nim_client


    # ========================================================
    # API SUMMARY
    # ========================================================

    @staticmethod
    def _api_summary(x):

        keys = (

            "api_key",

            "api_name",

            "block_type",

            "purpose",

            "inputs",

            "parameters",

            "default_values",

            "status",

            "candidate_reason",
        )

        return {

            key: x.get(key)

            for key in keys

            if x.get(key) is not None
        }


    # ========================================================
    # SIGNAL SUMMARY
    # ========================================================

    @staticmethod
    def _signal_summary(x):

        return {

            "signal_key":
                x.get(
                    "signal_key"
                ),

            "tams_id":
                x.get(
                    "TAMS ID"
                ),

            "model_path":
                x.get(
                    "Model Path"
                ),

            "type":
                x.get(
                    "Type"
                ),

            "description":
                x.get(
                    "Description"
                ),

            "status":
                x.get(
                    "status"
                ),
        }


    # ========================================================
    # NIM PLANNER PAYLOAD
    # ========================================================

    def _planner_payload(
        self,
        s41
    ):

        return {

            "task":
                "Create an SD generation PLAN only. "
                "Do not generate Python.",


            "planning_rules": {

                "candidate_api_rule":

                    "Candidate APIs may be proposed "
                    "and their known schemas may be "
                    "used for planning, but they must "
                    "remain candidate_requires_approval.",


                "metadata_rule":

                    "@platform and @TAMS are "
                    "step-definition metadata/decorators, "
                    "not API parameters.",


                "default_rule":

                    "Use supplied API default_values "
                    "as proposed defaults instead of "
                    "replacing them with TBD.",


                "approval_rule":

                    "Stage 5 never authorizes code "
                    "generation. ready_for_generation "
                    "must remain false until explicit "
                    "user approval outside this "
                    "planning step.",
            },


            "required_output_schema": {

                "summary":
                    "string",

                "proposed_keyword_pattern":
                    "string or null",

                "step_type":
                    "given|when|then|null",

                "selected_signals": [
                    "catalogue signal keys only"
                ],

                "proposed_apis": [

                    {
                        "api_key":
                            "catalogue key",

                        "status":
                            "confirmed|"
                            "candidate_requires_approval",

                        "reason":
                            "string",
                    }
                ],

                "parameter_plan": [

                    {
                        "api_key":
                            "string",

                        "parameters": {

                            "catalogue parameter":
                                "source/value/TBD"
                        },
                    }
                ],

                "execution_flow": [
                    "ordered planning steps"
                ],

                "questions": [

                    {
                        "id":
                            "stable_id",

                        "question":
                            "string",

                        "required":
                            True,

                        "reason":
                            "string",
                    }
                ],

                "ready_for_approval":
                    "boolean",

                "ready_for_generation":
                    "must always be false "
                    "in Stage 5",
            },


            "stage41": {

                "query":
                    s41[
                        "query"
                    ],

                "keyword_status":
                    s41[
                        "keyword_status"
                    ],

                "exact_existing_keyword":
                    s41[
                        "exact_existing_keyword"
                    ],

                "best_matching_keyword":
                    s41[
                        "best_matching_keyword"
                    ],

                "known_signals": [

                    self._signal_summary(
                        x
                    )

                    for x in
                    s41[
                        "known_signals"
                    ]
                ],

                "confirmed_apis": [

                    self._api_summary(
                        x
                    )

                    for x in
                    s41[
                        "confirmed_apis"
                    ]
                ],

                "candidate_apis": [

                    self._api_summary(
                        x
                    )

                    for x in
                    s41[
                        "candidate_apis"
                    ]
                ],

                "missing_information":
                    s41[
                        "missing_information"
                    ],

                "similar_sd_files":
                    s41[
                        "similar_sd_files"
                    ],

                "guardrails":
                    s41[
                        "generation_guardrails"
                    ],
            },
        }


    # ========================================================
    # QUESTION HELPER
    # ========================================================

    @staticmethod
    def _q(
        qid,
        question,
        reason
    ):

        return {

            "id":
                qid,

            "question":
                question,

            "required":
                True,

            "reason":
                reason,
        }


    # ========================================================
    # PARAMETER HELPERS
    # ========================================================

    @staticmethod
    def _parameter_names(
        api_record
    ):
        """
        Prefer typed parameters such as:

            String.Variable
            Variant.value
            Float.PostDelay

        Fall back to inputs only if typed
        parameters are unavailable.
        """

        typed_parameters = (
            api_record.get(
                "parameters"
            )
            or {}
        )

        if typed_parameters:

            return list(
                typed_parameters.keys()
            )


        return list(
            api_record.get(
                "inputs"
            )
            or []
        )


    @staticmethod
    def _parameter_base_name(
        parameter_name
    ):

        if "." in parameter_name:

            return parameter_name.split(
                ".",
                1
            )[1]

        return parameter_name


    @classmethod
    def _build_parameter_plan(
        cls,
        api_record,
        default_message=(
            "TBD from keyword/"
            "user confirmation"
        ),
    ):

        parameters = {}

        defaults = (
            api_record.get(
                "default_values"
            )
            or {}
        )


        for parameter_name in (
            cls._parameter_names(
                api_record
            )
        ):

            base_name = (
                cls._parameter_base_name(
                    parameter_name
                )
            )


            if base_name in defaults:

                parameters[
                    parameter_name
                ] = (
                    f"{defaults[base_name]} "
                    "(provided default)"
                )


            elif parameter_name in defaults:

                parameters[
                    parameter_name
                ] = (
                    f"{defaults[parameter_name]} "
                    "(provided default)"
                )


            else:

                parameters[
                    parameter_name
                ] = default_message


        return parameters


    # ========================================================
    # OFFLINE PLANNER
    # ========================================================

    def _offline_plan(
        self,
        s41
    ):

        status = (
            s41[
                "keyword_status"
            ]
        )


        signals = [

            x[
                "signal_key"
            ]

            for x in
            s41[
                "known_signals"
            ]
        ]


        candidates = [

            x[
                "api_key"
            ]

            for x in
            s41[
                "candidate_apis"
            ]
        ]


        questions = []

        apis = []

        parameter_plan = []

        flow = []


        # ====================================================
        # EXACT EXISTING
        # ====================================================

        if status == "EXACT_EXISTING":

            best = (
                s41[
                    "best_matching_keyword"
                ]
                or {}
            )


            for api in (
                s41[
                    "confirmed_apis"
                ]
            ):

                apis.append({

                    "api_key":
                        api[
                            "api_key"
                        ],

                    "status":
                        "confirmed",

                    "reason":
                        "observed in exact "
                        "existing SD "
                        "implementation",
                })


                parameter_plan.append({

                    "api_key":
                        api[
                            "api_key"
                        ],

                    "parameters": {

                        parameter:
                            "reuse observed "
                            "exact-SD mapping"

                        for parameter in
                        self._parameter_names(
                            api
                        )
                    },
                })


            flow = [

                "Reuse the exact existing "
                "SD implementation pattern",

                "Bind query values to "
                "existing Behave placeholders",

                "Preserve catalogue-backed "
                "API and signal references",
            ]


            return {

                "summary":

                    "Existing keyword implementation "
                    "found; the known implementation "
                    "can be prepared for user review.",


                "proposed_keyword_pattern":

                    best.get(
                        "pattern"
                    ),


                "step_type":

                    best.get(
                        "step_type"
                    ),


                "selected_signals":

                    signals,


                "proposed_apis":

                    apis,


                "parameter_plan":

                    parameter_plan,


                "execution_flow":

                    flow,


                "questions":

                    [],


                "ready_for_approval":

                    True,


                "ready_for_generation":

                    False,
            }


        # ====================================================
        # NEW WITH KNOWN CATALOGUE
        # ====================================================

        if (
            status
            == "NEW_WITH_KNOWN_CATALOGUE"
        ):

            for api in (
                s41[
                    "candidate_apis"
                ][:2]
            ):

                apis.append({

                    "api_key":

                        api[
                            "api_key"
                        ],


                    "status":

                        "candidate_requires_approval",


                    "reason":

                        api.get(
                            "candidate_reason",
                            "Stage 4.1 candidate",
                        ),
                })


                parameter_plan.append({

                    "api_key":

                        api[
                            "api_key"
                        ],


                    "parameters":

                        self._build_parameter_plan(
                            api
                        ),
                })


            questions.append(

                self._q(

                    "confirm_api",

                    "Confirm implementation API. "
                    "Candidate(s): "
                    f"{', '.join(candidates) or 'none'}.",

                    "New keywords cannot "
                    "promote candidate APIs "
                    "automatically.",
                )
            )


            questions.append(

                self._q(

                    "platform_tams",

                    "Provide/confirm @platform "
                    "and @TAMS decorator values.",

                    "These are step-definition "
                    "metadata and are not "
                    "API parameters.",
                )
            )


            flow = [

                "Use confirmed Signal "
                "Catalogue reference",

                "Use only a user-approved "
                "Function Catalogue "
                "candidate API",

                "Map keyword values to "
                "known API parameters",

                "Use supplied catalogue/"
                "default values where available",
            ]


            return {

                "summary":

                    "New keyword with known "
                    "catalogue signal; "
                    "implementation choices "
                    "require user input/approval.",


                "proposed_keyword_pattern":

                    s41[
                        "query"
                    ],


                "step_type":

                    None,


                "selected_signals":

                    signals,


                "proposed_apis":

                    apis,


                "parameter_plan":

                    parameter_plan,


                "execution_flow":

                    flow,


                "questions":

                    questions,


                "ready_for_approval":

                    False,


                "ready_for_generation":

                    False,
            }


        # ====================================================
        # INSUFFICIENT KNOWLEDGE
        # ====================================================

        questions = [

            self._q(

                "signal_catalogue",

                "Provide the required Signal "
                "Catalogue key/TAMS ID/model "
                "path, or add it to the catalogue.",

                "No authoritative signal "
                "was identified.",
            ),


            self._q(

                "implementation_api",

                "Provide/confirm the Function "
                "Catalogue API to implement "
                "this behavior.",

                "No implementation API can "
                "be selected safely.",
            ),


            self._q(

                "platform_tams",

                "Provide/confirm @platform "
                "and @TAMS decorator values.",

                "Required step-definition "
                "metadata is unavailable.",
            ),
        ]


        return {

            "summary":

                "Knowledge is insufficient "
                "to create an implementation "
                "plan safely.",


            "proposed_keyword_pattern":

                s41[
                    "query"
                ],


            "step_type":

                None,


            "selected_signals":

                [],


            "proposed_apis":

                [],


            "parameter_plan":

                [],


            "execution_flow":

                [],


            "questions":

                questions,


            "ready_for_approval":

                False,


            "ready_for_generation":

                False,
        }


    # ========================================================
    # BUILD ALLOWED API SCHEMAS
    # ========================================================

    @classmethod
    def _build_allowed_api_schemas(
        cls,
        s41
    ):
        """
        Build parameter schemas for confirmed AND
        candidate APIs.

        IMPORTANT:

        Candidate presence here does NOT mean the
        candidate API has been approved.

        This lookup is used only for closed-list
        parameter validation.
        """

        allowed = {}


        # ----------------------------------------------------
        # Confirmed APIs
        # ----------------------------------------------------

        for api in (
            s41[
                "confirmed_apis"
            ]
        ):

            api_key = (
                api.get(
                    "api_key"
                )
            )

            if not api_key:
                continue


            allowed[
                api_key
            ] = {

                "approval_status":

                    "confirmed",


                "parameters":

                    set(
                        cls._parameter_names(
                            api
                        )
                    ),


                "default_values":

                    api.get(
                        "default_values"
                    )
                    or {},
            }


        # ----------------------------------------------------
        # Candidate APIs
        # ----------------------------------------------------

        for api in (
            s41[
                "candidate_apis"
            ]
        ):

            api_key = (
                api.get(
                    "api_key"
                )
            )

            if not api_key:
                continue


            allowed[
                api_key
            ] = {

                "approval_status":

                    "candidate_only",


                "parameters":

                    set(
                        cls._parameter_names(
                            api
                        )
                    ),


                "default_values":

                    api.get(
                        "default_values"
                    )
                    or {},
            }


        return allowed


    # ========================================================
    # VALIDATOR
    # ========================================================

    def _validate(
        self,
        plan,
        s41
    ):

        errors = []


        # ----------------------------------------------------
        # Defensive response validation
        # ----------------------------------------------------

        if not isinstance(
            plan,
            dict
        ):

            return [

                "Stage 5 plan must be "
                "a JSON object/dictionary."
            ]


        # ----------------------------------------------------
        # Allowed signals
        # ----------------------------------------------------

        allowed_signals = {

            x[
                "signal_key"
            ]

            for x in
            s41[
                "known_signals"
            ]
        }


        # ----------------------------------------------------
        # Confirmed APIs
        # ----------------------------------------------------

        confirmed = {

            x[
                "api_key"
            ]

            for x in
            s41[
                "confirmed_apis"
            ]
        }


        # ----------------------------------------------------
        # Candidate APIs
        # ----------------------------------------------------

        candidate = {

            x[
                "api_key"
            ]

            for x in
            s41[
                "candidate_apis"
            ]
        }


        allowed_apis = (
            confirmed
            |
            candidate
        )


        api_schemas = (
            self._build_allowed_api_schemas(
                s41
            )
        )


        # ====================================================
        # VALIDATE SIGNALS
        # ====================================================

        selected_signals = (
            plan.get(
                "selected_signals"
            )
            or []
        )


        if not isinstance(
            selected_signals,
            list
        ):

            errors.append(
                "selected_signals must "
                "be a list."
            )

            selected_signals = []


        for signal_key in (
            selected_signals
        ):

            if (
                signal_key
                not in allowed_signals
            ):

                errors.append(

                    "Plan invented/unapproved "
                    "signal key: "
                    f"{signal_key}"
                )


        # ====================================================
        # VALIDATE PROPOSED APIs
        # ====================================================

        proposed_apis = (
            plan.get(
                "proposed_apis"
            )
            or []
        )


        if not isinstance(
            proposed_apis,
            list
        ):

            errors.append(
                "proposed_apis must "
                "be a list."
            )

            proposed_apis = []


        for item in proposed_apis:

            if not isinstance(
                item,
                dict
            ):

                errors.append(
                    "Each proposed API "
                    "must be an object."
                )

                continue


            api_key = (
                item.get(
                    "api_key"
                )
            )


            if (
                api_key
                not in allowed_apis
            ):

                errors.append(

                    "Plan invented/unapproved "
                    "API key: "
                    f"{api_key}"
                )

                continue


            # Candidate APIs may NOT be promoted
            # automatically by NIM.
            if (

                s41[
                    "keyword_status"
                ]
                != "EXACT_EXISTING"

                and

                api_key
                in candidate

                and

                item.get(
                    "status"
                )
                == "confirmed"
            ):

                errors.append(

                    "Candidate API promoted "
                    "without approval: "
                    f"{api_key}"
                )


        # ====================================================
        # VALIDATE PARAMETER PLAN
        # ====================================================

        parameter_plan = (
            plan.get(
                "parameter_plan"
            )
            or []
        )


        if not isinstance(
            parameter_plan,
            list
        ):

            errors.append(
                "parameter_plan must "
                "be a list."
            )

            parameter_plan = []


        for item in (
            parameter_plan
        ):

            if not isinstance(
                item,
                dict
            ):

                errors.append(
                    "Each parameter_plan "
                    "entry must be an object."
                )

                continue


            api_key = (
                item.get(
                    "api_key"
                )
            )


            parameters_obj = (
                item.get(
                    "parameters"
                )
                or {}
            )


            if not isinstance(
                parameters_obj,
                dict
            ):

                errors.append(

                    "Parameter mapping for API "
                    f"{api_key} must be an object."
                )

                continue


            planned_parameters = set(
                parameters_obj.keys()
            )


            schema = (
                api_schemas.get(
                    api_key
                )
            )


            if not schema:

                errors.append(

                    "Parameter plan references "
                    "unapproved API: "
                    f"{api_key}"
                )

                continue


            allowed_parameters = (
                schema[
                    "parameters"
                ]
            )


            invented_parameters = (
                planned_parameters
                -
                allowed_parameters
            )


            for parameter_name in sorted(
                invented_parameters
            ):

                errors.append(

                    "Plan invented parameter "
                    f"{parameter_name} "
                    f"for API {api_key}"
                )


        # ====================================================
        # STATUS GUARDRAILS
        # ====================================================

        if (

            s41[
                "keyword_status"
            ]
            == "INSUFFICIENT_KNOWLEDGE"

            and

            plan.get(
                "ready_for_generation"
            )
        ):

            errors.append(

                "Insufficient knowledge "
                "cannot be ready for generation."
            )


        if (

            s41[
                "keyword_status"
            ]
            == "NEW_WITH_KNOWN_CATALOGUE"

            and

            candidate

            and

            not plan.get(
                "questions"
            )
        ):

            errors.append(

                "New keyword candidate API "
                "requires at least one "
                "confirmation question."
            )


        # ====================================================
        # STAGE 5 MUST NEVER AUTHORIZE GENERATION
        # ====================================================

        if plan.get(
            "ready_for_generation"
        ):

            errors.append(

                "Stage 5 cannot authorize "
                "final code generation before "
                "explicit user approval."
            )


        return errors


    # ========================================================
    # APPROVAL GATE
    # ========================================================

    @staticmethod
    def _apply_approval_gate(
        plan,
        validation_errors
    ):
        """
        Deterministic approval gate.

        Do NOT trust NIM to authorize Stage 6.
        """

        # Stage 5 itself NEVER directly
        # enables generation.
        plan[
            "ready_for_generation"
        ] = False


        # ----------------------------------------------------
        # Validation failure
        # ----------------------------------------------------

        if validation_errors:

            plan[
                "ready_for_approval"
            ] = False


            return {

                "status":
                    "BLOCKED_BY_VALIDATION",

                "required_before_stage6":
                    True,
            }


        # ----------------------------------------------------
        # Find required unanswered questions
        # ----------------------------------------------------

        questions = (
            plan.get(
                "questions"
            )
            or []
        )


        required_questions = [

            q

            for q in questions

            if (
                isinstance(
                    q,
                    dict
                )
                and
                q.get(
                    "required",
                    False
                )
            )
        ]


        # ----------------------------------------------------
        # User information still required
        # ----------------------------------------------------

        if required_questions:

            plan[
                "ready_for_approval"
            ] = False


            return {

                "status":
                    "PENDING_USER_INPUT",

                "required_before_stage6":
                    True,
            }


        # ----------------------------------------------------
        # Plan complete, waiting for explicit approval
        # ----------------------------------------------------

        plan[
            "ready_for_approval"
        ] = True


        return {

            "status":
                "PENDING_APPROVAL",

            "required_before_stage6":
                True,
        }


    # ========================================================
    # MAIN STAGE 5 ENTRY POINT
    # ========================================================

    def plan(
        self,
        query: str,
        mode="offline",
        top_files=3,
        candidate_k=40,
    ):

        # ----------------------------------------------------
        # Stage 4.1
        # ----------------------------------------------------

        s41 = (
            self.stage41.query(

                query,

                top_files=
                    top_files,

                candidate_k=
                    candidate_k,
            )
        )


        # ----------------------------------------------------
        # Build NIM payload
        # ----------------------------------------------------

        payload = (
            self._planner_payload(
                s41
            )
        )


        # ====================================================
        # NVIDIA NIM MODE
        # ====================================================

        if mode == "nim":

            if not self.nim_client:

                raise ValueError(

                    "NIM mode requested but "
                    "NIM client is not configured"
                )


            plan = (
                self.nim_client.create_plan(
                    payload
                )
            )


            provider = (
                "nvidia-nim"
            )


        # ====================================================
        # OFFLINE MODE
        # ====================================================

        elif mode == "offline":

            plan = (
                self._offline_plan(
                    s41
                )
            )


            provider = (
                "deterministic-offline"
            )


        # ====================================================
        # INVALID MODE
        # ====================================================

        else:

            raise ValueError(

                "mode must be "
                "'offline' or 'nim'"
            )


        # ====================================================
        # Defensive plan type check
        # ====================================================

        if not isinstance(
            plan,
            dict
        ):

            raise ValueError(

                "Stage 5 planner returned "
                "an invalid plan type. "
                "Expected dict, got "
                f"{type(plan).__name__}."
            )


        # Deterministic pattern guard: never let NIM generalize fixed nouns.
        # Preserve the AI's suggestion separately for audit/debugging.
        original_pattern = plan.get("proposed_keyword_pattern")
        safe_pattern = _safe_gherkin_pattern(
            query, original_pattern, s41.get("keyword_status", "")
        )
        if original_pattern != safe_pattern:
            plan["original_ai_keyword_pattern"] = original_pattern
            plan["proposed_keyword_pattern"] = safe_pattern
            plan["keyword_pattern_corrected"] = True

        # ====================================================
        # CLOSED-LIST VALIDATION
        # ====================================================

        validation_errors = (
            self._validate(
                plan,
                s41
            )
        )


        # ====================================================
        # DETERMINISTIC APPROVAL GATE
        # ====================================================

        approval = (
            self._apply_approval_gate(
                plan,
                validation_errors
            )
        )


        # ====================================================
        # FINAL STAGE 5 RESULT
        # ====================================================

        return {

            "schema_version":
                "stage5-v1",


            "query":
                query,


            "planner_mode":
                mode,


            "planner_provider":
                provider,


            "stage41_status":
                s41[
                    "keyword_status"
                ],


            "plan":
                plan,


            "validation": {

                "valid":
                    not validation_errors,

                "errors":
                    validation_errors,

                "closed_list_enforced":
                    True,
            },


            "approval":
                approval,


            "stage41_evidence":
                s41,
        }
