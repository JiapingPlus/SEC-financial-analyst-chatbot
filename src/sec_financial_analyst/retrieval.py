"""Metadata-aware retrieval over the local SEC filing index."""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")


@dataclass(frozen=True)
class QueryFilters:
    ticker: str | None = None
    year: int | None = None

    def as_metadata(self) -> dict[str, str | int]:
        result: dict[str, str | int] = {}
        if self.ticker:
            result["ticker"] = self.ticker
        if self.year:
            result["year"] = self.year
        return result


def extract_filters(
    query: str,
    aliases: dict[str, set[str]],
) -> QueryFilters:
    folded = query.casefold()
    matches = {
        ticker
        for ticker, names in aliases.items()
        if any(re.search(rf"\b{re.escape(name.casefold())}\b", folded) for name in names)
    }
    years = {int(value) for value in YEAR_PATTERN.findall(query)}
    return QueryFilters(
        ticker=next(iter(matches)) if len(matches) == 1 else None,
        # A comparison needs candidates from every named year, so only apply a
        # hard year filter when the query names exactly one year.
        year=next(iter(years)) if len(years) == 1 else None,
    )


class SecRetriever:
    def __init__(self, vectorstore: FAISS, *, top_k: int = 5) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.aliases = _aliases_from_index(vectorstore)

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(sorted(self.aliases))

    @property
    def corpus_size(self) -> int:
        return int(self.vectorstore.index.ntotal)

    def invoke(self, query: str) -> list[Document]:
        if not query.strip():
            raise ValueError("query must not be blank")
        metadata_filter = extract_filters(query, self.aliases).as_metadata()
        if not metadata_filter:
            return self.vectorstore.similarity_search(query, k=self.top_k)
        fetch_k = max(self.top_k, self.corpus_size)
        return self.vectorstore.similarity_search(
            query,
            k=self.top_k,
            filter=metadata_filter,
            fetch_k=fetch_k,
        )


def _aliases_from_index(vectorstore: FAISS) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}
    documents = getattr(vectorstore.docstore, "_dict", {}).values()
    for document in documents:
        ticker = str(document.metadata.get("ticker", "")).upper()
        if not ticker:
            continue
        names = aliases.setdefault(ticker, {ticker.casefold()})
        company_name = str(document.metadata.get("company_name", "")).strip()
        if company_name:
            names.add(company_name.casefold())
            base_name = re.sub(
                r"(?i)\s+(inc\.?|corp\.?|corporation|company|plc)$", "", company_name
            ).strip()
            if base_name:
                names.add(base_name.casefold())
    return aliases
