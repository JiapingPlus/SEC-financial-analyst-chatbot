"""Environment-driven configuration with public-repository-safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    sec_user_agent: str | None
    chat_model: str
    embedding_model: str
    sec_requests_per_second: float
    retriever_top_k: int
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    raw_dir: Path
    normalized_dir: Path
    processed_dir: Path
    manifest_dir: Path
    vectorstore_dir: Path

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Settings":
        load_dotenv(env_file or PROJECT_ROOT / ".env")
        data_dir = PROJECT_ROOT / "data"
        settings = cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            sec_user_agent=os.getenv("SEC_USER_AGENT"),
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
            embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            sec_requests_per_second=float(
                os.getenv("SEC_REQUESTS_PER_SECOND", "4")
            ),
            retriever_top_k=int(os.getenv("RETRIEVER_TOP_K", "5")),
            chunk_size_tokens=int(os.getenv("CHUNK_SIZE_TOKENS", "700")),
            chunk_overlap_tokens=int(os.getenv("CHUNK_OVERLAP_TOKENS", "80")),
            raw_dir=data_dir / "raw",
            normalized_dir=data_dir / "normalized",
            processed_dir=data_dir / "processed",
            manifest_dir=data_dir / "manifests",
            vectorstore_dir=data_dir / "vectorstore" / "sec_10k_faiss",
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not 0 < self.sec_requests_per_second <= 10:
            raise ValueError("SEC_REQUESTS_PER_SECOND must be between 0 and 10")
        if self.retriever_top_k <= 0:
            raise ValueError("RETRIEVER_TOP_K must be positive")
        if self.chunk_size_tokens <= 0:
            raise ValueError("CHUNK_SIZE_TOKENS must be positive")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than chunk size")

    def require_openai_key(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        return self.openai_api_key

    def require_sec_user_agent(self) -> str:
        if not self.sec_user_agent or "@" not in self.sec_user_agent:
            raise RuntimeError(
                "SEC_USER_AGENT must identify your project and include a contact email"
            )
        return self.sec_user_agent
