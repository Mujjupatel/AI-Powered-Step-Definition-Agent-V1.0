# Stage 7.2 — Existing Pattern / Missing Decorator Enhancement

## Files modified
- `retrieval/stage41.py`: detect same parameterized keyword with a missing Given/When/Then decorator; return `EXISTING_PATTERN_MISSING_DECORATOR` and source evidence. Do not promote observed action APIs to confirmed verification APIs.
- `ui/workflow.py`: retain Gherkin step type in feature-file analysis.
- `stage72_chat_workspace.py`: display extension advice, suggest verification candidates, block setter-based Then generation.

## Scope / safety
This build DOES NOT automatically modify an existing SD or add `@then` to an action implementation. It supports the **safe separate-implementation review path** in the existing approval/generation workflow. Editing an existing SD in place needs source-path, AST-aware patching, duplicate decorator checks, and a separate engineer-approved write action; those are not implemented here.

Existing index, catalogue, example_output, NVIDIA settings and model files should be copied from your current project; no retraining needed. Keep secrets and proprietary data out of shared archives.

Run `streamlit run stage72_chat_workspace.py` after merging with existing assets.
