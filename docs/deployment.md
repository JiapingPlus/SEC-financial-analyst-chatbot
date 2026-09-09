# Streamlit Community Cloud deployment

The public demo uses the repository's portable SEC index (`index.faiss` plus ordered JSONL
documents). It does not download filings, generate embeddings for the corpus, or deserialize
pickle during startup.

## Deploy

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with the GitHub account
   that administers this repository.
2. Choose **Create app** and enter:
   - Repository: `JiapingPlus/SEC-financial-analyst-chatbot`
   - Branch: `main`
   - Main file path: `app/dashboard.py`
   - App URL: an available portfolio-friendly subdomain
3. Open **Advanced settings**, select Python 3.12, and paste the values from
   `.streamlit/secrets.toml.example` after replacing the OpenAI placeholder with a real key.
4. Deploy, then ask one of the example questions and confirm that answer citations expand to
   direct SEC filing links.

## Cost controls

Public mode limits question length, questions per browser session, and total questions handled
by one app process. These are safeguards, not authentication. Monitor OpenAI usage and set an
account-level project budget before sharing the URL publicly. Use a dedicated OpenAI project key
for this demo so it can be rotated or disabled independently.

## Updates

Community Cloud watches `main`. A pushed code or dependency change triggers an app update;
dependency-file changes trigger a full rebuild.
