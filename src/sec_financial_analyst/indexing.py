"""Section-aware chunking and trusted local FAISS persistence."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings


KNOWN_HEADING = re.compile(
    r"(?i)^(?:item\s+7\b|management.{0,40}discussion|overview|"
    r"results? of operations|liquidity and capital resources|"
    r"critical accounting|business trends?|segment results?)"
)


def load_normalized_documents(path: Path) -> list[Document]:
    documents: list[Document] = []
    for file_path in sorted(path.glob("*.json")):
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        text = str(payload.get("text", "")).strip()
        metadata = payload.get("metadata", {})
        if text and isinstance(metadata, dict):
            documents.append(
                Document(
                    page_content=text,
                    metadata={**metadata, "source_file": file_path.name},
                )
            )
    if not documents:
        raise ValueError(f"No normalized SEC documents found in {path}")
    return documents


def split_sections(text: str) -> list[tuple[str, str]]:
    title = "Item 7 — Management's Discussion and Analysis"
    sections: list[tuple[str, str]] = []
    body: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if KNOWN_HEADING.search(line) and len(line) <= 180:
            if body:
                sections.append((title, "\n".join(body)))
                body = []
            title = line
        else:
            body.append(line)
    if body:
        sections.append((title, "\n".join(body)))
    return sections or [(title, text.strip())]


def chunk_documents(
    documents: Iterable[Document],
    *,
    chunk_size_tokens: int = 700,
    chunk_overlap_tokens: int = 80,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    chunks: list[Document] = []
    for document_index, document in enumerate(documents):
        for section_title, section_text in split_sections(document.page_content):
            for part in splitter.split_text(section_text):
                chunks.append(
                    Document(
                        page_content=f"Section: {section_title}\n\n{part.strip()}",
                        metadata={
                            **document.metadata,
                            "section_title": section_title,
                            "document_index": document_index,
                            "chunk_id": len(chunks),
                            "chunking_strategy": "section_token_recursive_v1",
                        },
                    )
                )
    return chunks


def build_index(settings: Settings) -> dict[str, Any]:
    documents = load_normalized_documents(settings.normalized_dir)
    chunks = chunk_documents(
        documents,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = settings.processed_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {"page_content": chunk.page_content, "metadata": chunk.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )
    embeddings = OpenAIEmbeddings(
        api_key=settings.require_openai_key(), model=settings.embedding_model
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(settings.vectorstore_dir))
    settings.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "documents": len(documents),
        "chunks": len(chunks),
        "embedding_model": settings.embedding_model,
        "chunk_size_tokens": settings.chunk_size_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "vectorstore": str(settings.vectorstore_dir),
    }
    (settings.manifest_dir / "index.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def load_index(settings: Settings) -> FAISS:
    if not settings.vectorstore_dir.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {settings.vectorstore_dir}; run build-index"
        )
    embeddings = OpenAIEmbeddings(
        api_key=settings.require_openai_key(), model=settings.embedding_model
    )
    # The index uses pickle-backed LangChain metadata. Only load an index built
    # locally by this project; never point this function at an untrusted download.
    return FAISS.load_local(
        str(settings.vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
