from dataclasses import dataclass

from langchain_core.documents import Document

from sec_financial_analyst.chatbot import (
    FinancialAnalystChatbot,
    Turn,
    is_sec_financial_question,
)
from sec_financial_analyst.retrieval import extract_filters


class StubRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.documents


@dataclass
class StubResponse:
    content: str


class StubModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def invoke(self, _messages):
        return StubResponse(next(self.responses))


def evidence_document() -> Document:
    return Document(
        page_content="Revenue grew because services sales increased.",
        metadata={
            "ticker": "AAPL",
            "filing_type": "10-K",
            "year": 2025,
            "section_title": "Products and Services Performance",
            "chunk_id": 7,
            "source_url": "https://www.sec.gov/example.htm",
        },
    )


def test_follow_up_is_rewritten_before_retrieval() -> None:
    retriever = StubRetriever([evidence_document()])
    model = StubModel(
        [
            "What affected Apple's revenue in fiscal 2024?",
            "Services supported revenue growth [1].",
        ]
    )
    chatbot = FinancialAnalystChatbot(retriever, model)

    response = chatbot.ask(
        "What about the previous year?",
        [Turn("What affected Apple's revenue in 2025?", "Services grew [1].")],
    )

    assert retriever.queries == ["What affected Apple's revenue in fiscal 2024?"]
    assert response.standalone_question.endswith("2024?")
    assert response.sources[0].label == "AAPL | 10-K | 2025 | chunk 7"


def test_out_of_scope_question_does_not_retrieve() -> None:
    retriever = StubRetriever([])
    chatbot = FinancialAnalystChatbot(retriever, StubModel([]))
    response = chatbot.ask("Write me a poem about the ocean")
    assert response.out_of_scope is True
    assert response.sources == ()
    assert retriever.queries == []


def test_scope_and_metadata_helpers() -> None:
    assert is_sec_financial_question("What risks appear in Apple's 10-K?")
    assert not is_sec_financial_question("How is the weather?")
    filters = extract_filters(
        "What drove Apple revenue in 2025?",
        {"AAPL": {"aapl", "apple", "apple inc."}},
    )
    assert filters.ticker == "AAPL"
    assert filters.year == 2025

    comparison = extract_filters(
        "Compare Apple revenue in 2024 and 2025",
        {"AAPL": {"aapl", "apple", "apple inc."}},
    )
    assert comparison.ticker == "AAPL"
    assert comparison.year is None
