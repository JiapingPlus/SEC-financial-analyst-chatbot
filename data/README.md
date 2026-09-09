# Generated data

This directory is intentionally data-light. The pipeline creates the following ignored
paths locally:

- `raw/` — downloaded SEC filing HTML
- `normalized/` — extracted Item 7 JSON records
- `processed/` — chunk-level JSONL used for inspection
- `vectorstore/` — local FAISS index and pickle-backed metadata
- `manifests/` — reproducibility summaries

Do not commit API keys, downloaded filing corpora, or FAISS pickle files. Rebuild them with
the documented CLI commands. SEC filing URLs and SHA-256 hashes remain in local manifests
for provenance.
