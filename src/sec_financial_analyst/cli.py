"""Command-line entry points for rebuilding and querying the pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .chatbot import build_chatbot
from .config import Settings
from .indexing import build_index
from .sec import SecClient


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="sec-analyst")
    commands = root.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download and normalize SEC 10-K Item 7")
    download.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA"])
    download.add_argument("--filings-per-ticker", type=int, default=3)
    download.add_argument("--plan-only", action="store_true")

    commands.add_parser("build-index", help="Chunk filings and build a local FAISS index")

    ask = commands.add_parser("ask", help="Ask a grounded single-turn SEC question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int)

    commands.add_parser("doctor", help="Check configuration and local artifacts")
    return root


def main() -> None:
    args = parser().parse_args()
    settings = Settings.from_env()
    if args.command == "download":
        client = SecClient(
            settings.require_sec_user_agent(),
            requests_per_second=settings.sec_requests_per_second,
        )
        filings = client.plan(args.tickers, args.filings_per_ticker)
        if args.plan_only:
            print(json.dumps([asdict(filing) for filing in filings], indent=2))
            return
        result = client.download_corpus(
            filings,
            raw_dir=settings.raw_dir,
            normalized_dir=settings.normalized_dir,
            manifest_dir=settings.manifest_dir,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "build-index":
        print(json.dumps(build_index(settings), indent=2))
        return
    if args.command == "ask":
        response = build_chatbot(settings, top_k=args.top_k).ask(args.question)
        print(json.dumps(asdict(response), ensure_ascii=False, indent=2))
        return
    status = {
        "openai_key_configured": bool(settings.openai_api_key),
        "sec_user_agent_configured": bool(settings.sec_user_agent),
        "normalized_filings": len(list(settings.normalized_dir.glob("*.json"))),
        "index_ready": settings.index_ready,
        "index_format": "portable" if settings.portable_index_dir.exists() else "local-faiss",
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
    }
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
