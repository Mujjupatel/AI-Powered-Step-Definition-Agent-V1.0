# Stage 8: Synthetic SD Agent cloud demo

## Verified project assets
- Entry point: `stage72_chat_workspace.py` (alternate older UI: `streamlit_app.py`).
- Catalogue: `example_output/catalogue_snapshot.json` (no `data/synthetic_catalogues` folder needed).
- RAG files: `rag_index/embeddings.npy`, `rag_index/metadata.jsonl`, `rag_index/index_manifest.json`, `rag_index/embedding_model` (a joblib-serialized TF-IDF vectorizer).
- Existing example knowledge: `example_output/knowledge_cards.json`, `example_output/rag_chunks.jsonl`.
- Python packages: `streamlit`, `numpy`, `scikit-learn`, `joblib`. NVIDIA NIM requests use Python standard library.

## Publish to GitHub
1. Unzip this archive into a clean local folder and review every file for synthetic-only content.
2. Copy these files into your GitHub repo, keeping the same relative directory structure.
3. Commit changes to `main`. Do **not** commit `.env`, `.streamlit/secrets.toml`, or API keys.
4. In Streamlit Community Cloud create an app from your repository, branch `main`, main file path `stage72_chat_workspace.py`.
5. Under app **Advanced settings / Secrets**, add:
   ```toml
   NVIDIA_API_KEY = "YOUR_DEMO_NVIDIA_KEY"
   NVIDIA_MODEL = "google/gemma-3-12b-it"
   ```
   Use a new demo key with spending/rate limits; never paste it in chat or GitHub.
6. Select a supported Python version in Cloud settings (Python 3.11 is a reasonable starting point), deploy, and inspect build logs.
7. Test on a mobile browser using the generated HTTPS `*.streamlit.app` URL.

## Important caveats
- **Public demo**: anyone with the URL may invoke NVIDIA requests and consume quota. Cloud deployment is not authentication or per-user rate limiting.
- **No persistent writes**: local updates to `data/approved_signals.json` can disappear on redeploy/restart. Multi-user writes may interfere. Treat registration as demo-only; avoid sensitive data.
- **Do not execute uploaded Python**: upload original SDs only as synthetic source for review/merge.
- **TF-IDF index**: `rag_index/embedding_model` is a joblib file. Only load it from your trusted project; never accept arbitrary model uploads. Version drift between sklearn releases can cause warnings or failure; if necessary rebuild the index with the deployed dependency versions.
- **Runtime not verified here**: Streamlit and NVIDIA NIM were unavailable in the packaging environment; only static checks and local retrieval smoke tests were possible.
- **Pending enhancements**: automatic mapping extraction and dynamic placeholder rename/binding remain future work.
- **HTTPS**: the cloud URL uses HTTPS; your earlier LAN IP access is normally plain HTTP.
