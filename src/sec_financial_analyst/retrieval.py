"""Metadata-aware retrieval over the local SEC filing index."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")


class QueryEmbeddings(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


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


class PortableSecRetriever:
    """Read a FAISS binary plus ordered JSONL documents without pickle."""

    def __init__(
        self,
        index: faiss.Index,
        documents: list[Document],
        embeddings: QueryEmbeddings,
        *,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if index.ntotal != len(documents):
            raise ValueError("Portable index and document counts do not match")
        self.index = index
        self.documents = documents
        self.embeddings = embeddings
        self.top_k = top_k
        self.aliases = _aliases_from_documents(documents)

    @classmethod
    def load(
        cls,
        path: Path,
        embeddings: QueryEmbeddings,
        *,
        top_k: int = 5,
        expected_embedding_model: str | None = None,
    ) -> "PortableSecRetriever":
        index_path = path / "index.faiss"
        documents_path = path / "documents.jsonl"
        if not index_path.is_file() or not documents_path.is_file():
            raise FileNotFoundError(f"Portable SEC index is incomplete at {path}")
        manifest_path = path / "manifest.json"
        if expected_embedding_model and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            actual_model = manifest.get("embedding_model")
            if actual_model and actual_model != expected_embedding_model:
                raise ValueError(
                    "Portable index embedding model does not match configuration: "
                    f"{actual_model!r} != {expected_embedding_model!r}"
                )
        documents: list[Document] = []
        for line in documents_path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            documents.append(
                Document(
                    page_content=str(payload["page_content"]),
                    metadata=dict(payload.get("metadata", {})),
                )
            )
        return cls(faiss.read_index(str(index_path)), documents, embeddings, top_k=top_k)

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(sorted(self.aliases))

    @property
    def corpus_size(self) -> int:
        return len(self.documents)

    def invoke(self, query: str) -> list[Document]:
        if not query.strip():
            raise ValueError("query must not be blank")
        filters = extract_filters(query, self.aliases).as_metadata()
        vector = np.asarray([self.embeddings.embed_query(query)], dtype="float32")
        if vector.shape[1] != self.index.d:
            raise ValueError(
                f"Query embedding has {vector.shape[1]} dimensions; index expects {self.index.d}"
            )
        _, positions = self.index.search(vector, self.corpus_size)
        matches: list[Document] = []
        for position in positions[0]:
            if position < 0:
                continue
            document = self.documents[int(position)]
            if all(document.metadata.get(key) == value for key, value in filters.items()):
                matches.append(document)
            if len(matches) == self.top_k:
                break
        return matches


def _aliases_from_index(vectorstore: FAISS) -> dict[str, set[str]]:
    documents = getattr(vectorstore.docstore, "_dict", {}).values()
    return _aliases_from_documents(documents)


def _aliases_from_documents(documents) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}
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
