from langchain_core.documents import Document

from sec_financial_analyst.indexing import chunk_documents, split_sections


def test_split_sections_preserves_financial_headings() -> None:
    sections = split_sections(
        "Results of Operations\nRevenue increased because demand improved.\n"
        "Liquidity and Capital Resources\nCash funded capital expenditure."
    )
    assert [title for title, _ in sections] == [
        "Results of Operations",
        "Liquidity and Capital Resources",
    ]


def test_chunk_documents_adds_traceable_metadata() -> None:
    text = "Results of Operations\n" + ("Revenue increased year over year. " * 100)
    chunks = chunk_documents(
        [Document(page_content=text, metadata={"ticker": "TEST", "year": 2025})],
        chunk_size_tokens=80,
        chunk_overlap_tokens=10,
    )
    assert len(chunks) > 1
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.metadata["ticker"] == "TEST" for chunk in chunks)
    assert all(chunk.page_content.startswith("Section: Results of Operations") for chunk in chunks)
