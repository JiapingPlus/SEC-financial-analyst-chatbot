# SEC Financial Analyst Chatbot

A conversational retrieval-augmented generation application for analyzing SEC 10-K
Management's Discussion and Analysis (Item 7). It downloads filings from EDGAR, extracts and
chunks the relevant section, builds a local FAISS index, rewrites follow-up questions, and
returns evidence-grounded answers with inspectable citations.

> Portfolio project: the focus is transparent financial research engineering, not investment
> advice or real-time market prediction.

## Why this project

Financial answers are only useful when a reviewer can trace them back to the filing. This
project makes retrieval visible: every generated claim is grounded in numbered context, and
the dashboard exposes the exact SEC excerpts used by the model.

## Features

- End-to-end SEC EDGAR → Item 7 → chunks → embeddings → FAISS pipeline
- SEC-compliant user agent, rate limiting, retry behavior, local caching, and manifests
- Section-aware, token-budgeted chunking with filing and source metadata
- Metadata-aware company/year filtering before semantic retrieval
- Multi-turn follow-up rewriting (for example, “What about the previous year?”)
- SEC financial scope guard for unrelated questions
- Grounded generation with numbered citations and explicit insufficient-evidence behavior
- Streamlit analyst dashboard with chat history and expandable source cards
- Offline unit tests, environment diagnostics, and GitHub Actions CI
- Cloud-safe portable FAISS + JSONL demo index (no pickle deserialization at startup)
- Public-demo request budgets and Streamlit Community Cloud configuration

## Architecture

```text
SEC EDGAR -> 10-K Item 7 extraction -> section-aware chunks -> OpenAI embeddings
                                                            -> local FAISS index

conversation -> standalone-query rewrite -> scope guard -> metadata-aware retrieval
             -> grounded answer -> numbered citations + SEC evidence cards
```

See [the detailed architecture](docs/architecture.md).

## Quick start

Requirements: Python 3.11–3.13 and an OpenAI API key.

```powershell
git clone https://github.com/JiapingPlus/SEC-financial-analyst-chatbot.git
cd SEC-financial-analyst-chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Edit `.env` with:

- `OPENAI_API_KEY`
- `SEC_USER_AGENT`, including a real contact email as required for responsible EDGAR access

Build a small reproducible corpus:

```powershell
# Inspect the exact download plan first.
sec-analyst download --tickers AAPL MSFT NVDA --filings-per-ticker 3 --plan-only

# Download and extract 10-K Item 7 sections.
sec-analyst download --tickers AAPL MSFT NVDA --filings-per-ticker 3

# Chunk, embed, and build the local FAISS index.
sec-analyst build-index

# Verify configuration and artifacts.
sec-analyst doctor
```

Launch the dashboard:

```powershell
python -m streamlit run app\dashboard.py
```

## Online demo deployment

The repository is ready for Streamlit Community Cloud with `app/dashboard.py` as its entrypoint
and Python 3.12. Add `OPENAI_API_KEY` through the platform's Secrets settings—never commit it.
The included portable index lets the cloud app start immediately with SEC evidence for AAPL,
MSFT, TSLA, NVDA, and AMZN.

See [the deployment runbook](docs/deployment.md) for the exact fields, secrets template, test,
and cost-control checklist.

Or ask a single question from the CLI:

```powershell
sec-analyst ask "What factors affected Apple's revenue growth in fiscal 2025?"
```

## Good questions

- What factors affected Apple's revenue growth in fiscal 2025?
- What liquidity risks did Microsoft discuss in its latest indexed 10-K?
- Compare Nvidia's revenue drivers across the two latest indexed fiscal years.
- What did the company say about inflation, foreign exchange, or supply-chain pressure?

Questions about live prices, current news, personal portfolios, or trading recommendations are
outside the indexed SEC corpus and are rejected or answered with an evidence limitation.

## Repository map

```text
app/dashboard.py                    Streamlit analyst interface
.streamlit/config.toml              Portfolio dashboard theme
data/demo_index/                    Portable 521-chunk public-demo index
src/sec_financial_analyst/sec.py    EDGAR ingestion + Item 7 extraction
src/sec_financial_analyst/indexing.py  Section chunking + FAISS persistence
src/sec_financial_analyst/retrieval.py Metadata-aware retrieval
src/sec_financial_analyst/chatbot.py   Follow-up rewriting + grounded generation
src/sec_financial_analyst/cli.py       Reproducible pipeline commands
tests/                              Offline unit tests
docs/architecture.md                Design and trust boundaries
docs/deployment.md                  Streamlit Community Cloud runbook
```

## Data and security

- `.env`, raw filings, normalized corpora, generated chunks, manifests, and pickle-backed local
  indexes are ignored by Git.
- The repository never contains an API key.
- The bundled demo uses a non-pickle FAISS + JSONL format; the development pipeline loads its
  pickle-backed index only when it was built locally by this project.
- Retrieved filing text is treated as untrusted evidence and is HTML-escaped in the dashboard.
- Generated answers are research assistance, not investment advice.

## Tests

```powershell
python -m pytest
```

The tests run offline and cover SEC extraction, hidden XBRL removal, section chunking,
metadata filtering, scope control, follow-up rewriting, and exact source mapping.

## Roadmap

- Add retrieval and citation evaluation on a fixed SEC question set
- Add streamed answers and latency/token telemetry
- Add optional hybrid BM25 + dense retrieval

## License

MIT
