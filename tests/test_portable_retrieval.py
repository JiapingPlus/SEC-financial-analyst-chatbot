import json

import faiss
import numpy as np

from sec_financial_analyst.retrieval import PortableSecRetriever


class StubEmbeddings:
    def embed_query(self, _text: str) -> list[float]:
        return [1.0, 0.0]


def test_portable_retriever_loads_jsonl_and_filters_metadata(tmp_path) -> None:
    index = faiss.IndexFlatL2(2)
    index.add(np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype="float32"))
    faiss.write_index(index, str(tmp_path / "index.faiss"))
    payloads = [
        {
            "page_content": "Apple revenue grew.",
            "metadata": {"ticker": "AAPL", "company_name": "Apple Inc.", "year": 2025},
        },
        {
            "page_content": "Apple liquidity remained strong.",
            "metadata": {"ticker": "AAPL", "company_name": "Apple Inc.", "year": 2024},
        },
        {
            "page_content": "Microsoft cloud revenue grew.",
            "metadata": {"ticker": "MSFT", "company_name": "Microsoft Corp", "year": 2025},
        },
    ]
    (tmp_path / "documents.jsonl").write_text(
        "\n".join(json.dumps(item) for item in payloads) + "\n",
        encoding="utf-8",
    )

    retriever = PortableSecRetriever.load(tmp_path, StubEmbeddings(), top_k=2)
    results = retriever.invoke("What affected Apple revenue in 2025?")

    assert retriever.corpus_size == 3
    assert retriever.tickers == ("AAPL", "MSFT")
    assert [document.metadata["year"] for document in results] == [2025]


def test_portable_retriever_rejects_mismatched_document_count(tmp_path) -> None:
    index = faiss.IndexFlatL2(2)
    index.add(np.asarray([[1.0, 0.0]], dtype="float32"))
    faiss.write_index(index, str(tmp_path / "index.faiss"))
    (tmp_path / "documents.jsonl").write_text("", encoding="utf-8")

    try:
        PortableSecRetriever.load(tmp_path, StubEmbeddings())
    except ValueError as error:
        assert "counts do not match" in str(error)
    else:
        raise AssertionError("Expected mismatched portable index to fail")
