# Stage 6.5 safe SD merge patch

Copy `ui/sd_merge.py` and `stage72_chat_workspace.py` into the current working project. The included `ui/stage6_generator.py` is an unchanged reference copy and need not be replaced. Do NOT replace your entire project or index.

For a missing Then decorator, generate an approved check workflow. The workspace now creates both standalone output and a merged original SD preview when the full original source is available in retrieval evidence. The merged download uses the original filename. It never overwrites your project file.

Limitations: cross-file duplicate Behave registrations are not checked; only the target source is checked. AST rewriting affects only the new generated function. Generated code needs runtime testing against your real AutomationDesk catalogues. The current UI does not support uploading an original source if retrieval lacks it.
