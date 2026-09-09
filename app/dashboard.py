"""Streamlit portfolio dashboard for conversational SEC analysis."""

from __future__ import annotations

from html import escape

import streamlit as st

from sec_financial_analyst.chatbot import ChatResponse, Source, Turn, build_chatbot
from sec_financial_analyst.config import Settings


EXAMPLES = (
    "What drove Apple's revenue growth in fiscal 2025?",
    "What liquidity risks did Microsoft discuss in its latest indexed 10-K?",
    "Compare Nvidia's revenue drivers across the two latest indexed fiscal years.",
)

st.set_page_config(
    page_title="SEC Financial Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def cached_chatbot(top_k: int):
    return build_chatbot(top_k=top_k)


def styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1500px; padding-top: 2rem; padding-bottom: 3rem;}
        [data-testid="stSidebar"] {border-right: 1px solid rgba(125,145,165,.2);}
        .kicker {color:#2a9d8f; font-weight:700; letter-spacing:.14em;
                 text-transform:uppercase; font-size:.78rem;}
        .subtitle {color:#73808c; margin-top:-.45rem; margin-bottom:1rem;}
        .evidence {border-left:3px solid #2a9d8f; background:rgba(42,157,143,.07);
                   border-radius:0 .5rem .5rem 0; padding:.8rem 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def source_card(source: Source) -> None:
    with st.expander(f"[{source.rank}] {source.label}"):
        left, middle, right = st.columns(3)
        left.metric("Ticker", source.ticker)
        middle.metric("Fiscal year", source.year)
        right.metric("Evidence rank", source.rank)
        st.caption(source.section)
        st.markdown(
            f"<div class='evidence'>{escape(source.excerpt)}</div>",
            unsafe_allow_html=True,
        )
        if source.url:
            st.link_button("Open SEC filing ↗", source.url)


def assistant_message(response: ChatResponse) -> None:
    if response.out_of_scope:
        st.warning(response.answer)
    else:
        st.markdown(response.answer)
    if response.standalone_question != response.question:
        st.caption(f"Follow-up interpreted as: {response.standalone_question}")
    st.markdown(f"**Retrieved evidence · {len(response.sources)} sources**")
    st.caption(f"Completed in {response.elapsed_seconds:.1f}s")
    for source in response.sources:
        source_card(source)


def main() -> None:
    styles()
    settings = Settings.from_env()
    st.session_state.setdefault("messages", [])
    selected: str | None = None

    with st.sidebar:
        st.header("Analyst controls")
        default_top_k = min(max(settings.retriever_top_k, 2), 10)
        top_k = st.slider("Evidence budget", 2, 10, default_top_k)
        st.divider()
        st.subheader("System status")
        st.write(f"{'🟢' if settings.openai_api_key else '🔴'} OpenAI configuration")
        st.write(f"{'🟢' if settings.vectorstore_dir.exists() else '🔴'} Local SEC index")
        st.caption(f"Answer model: `{settings.chat_model}`")
        st.divider()
        st.subheader("Example questions")
        for index, question in enumerate(EXAMPLES, 1):
            if st.button(question, key=f"example-{index}", use_container_width=True):
                selected = question
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.caption("Grounded SEC research assistance — not investment advice.")

    st.markdown("<div class='kicker'>Conversational RAG · SEC 10-K</div>", unsafe_allow_html=True)
    st.title("SEC Financial Analyst")
    st.markdown(
        "<div class='subtitle'>Ask filing-grounded questions, continue with follow-ups, "
        "and inspect the exact evidence behind every answer.</div>",
        unsafe_allow_html=True,
    )
    first, second, third = st.columns(3)
    first.metric("Knowledge base", "SEC 10-K · Item 7")
    second.metric("Retrieval", "Metadata-aware FAISS")
    third.metric("Evidence budget", f"Top {top_k}")
    st.divider()

    for item in st.session_state.messages:
        with st.chat_message("user"):
            st.markdown(item["question"])
        with st.chat_message("assistant"):
            assistant_message(item["response"])

    typed = st.chat_input("Ask about revenue, margins, liquidity, risks, or segments…")
    question = selected or typed
    if not question:
        return
    history = [
        Turn(question=item["question"], answer=item["response"].answer)
        for item in st.session_state.messages
    ]
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Interpreting the conversation and searching SEC filings…"):
                response = cached_chatbot(top_k).ask(question, history)
            assistant_message(response)
            st.session_state.messages.append({"question": question, "response": response})
        except Exception as exc:
            st.error(f"Analysis unavailable: {exc}")
            st.info("Run `sec-analyst doctor`, then download filings and build the index.")


if __name__ == "__main__":
    main()
