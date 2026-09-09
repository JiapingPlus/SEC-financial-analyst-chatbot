"""SEC EDGAR ingestion and resilient 10-K Item 7 extraction."""

from __future__ import annotations

import hashlib
import json
import re
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning


TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
ITEM_7_START = re.compile(r"(?im)^[ \t]*item[\s\u00a0]+7(?![a-z0-9])[^\n]{0,180}")
ITEM_7_END = re.compile(
    r"(?im)^[ \t]*item[\s\u00a0]+(?:7[\s\u00a0]*a|8)(?![a-z0-9])[^\n]{0,180}"
)
MDA_TITLE = re.compile(r"(?i)management.{0,40}discussion.{0,60}analysis")


class SecError(RuntimeError):
    """Raised for SEC transport, metadata, or extraction failures."""


@dataclass(frozen=True)
class Filing:
    ticker: str
    company_name: str
    cik: str
    accession_number: str
    filing_date: str
    report_date: str
    primary_document: str

    @property
    def year(self) -> int:
        return int(self.report_date[:4])

    @property
    def source_url(self) -> str:
        accession = self.accession_number.replace("-", "")
        return (
            f"{ARCHIVES_URL}/{int(self.cik)}/{accession}/{self.primary_document}"
        )


class SecClient:
    """Small SEC client that follows the published 10 requests/second ceiling."""

    def __init__(
        self,
        user_agent: str,
        *,
        requests_per_second: float = 4,
        timeout_seconds: float = 45,
        session: requests.Session | None = None,
    ) -> None:
        if "@" not in user_agent:
            raise ValueError("SEC user agent must include a contact email")
        if not 0 < requests_per_second <= 10:
            raise ValueError("requests_per_second must be between 0 and 10")
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
            }
        )
        self.minimum_interval = 1 / requests_per_second
        self.timeout_seconds = timeout_seconds
        self.last_request_at: float | None = None

    def get(self, url: str) -> requests.Response:
        for attempt in range(1, 5):
            now = time.monotonic()
            if self.last_request_at is not None:
                time.sleep(max(0, self.minimum_interval - (now - self.last_request_at)))
            self.last_request_at = time.monotonic()
            try:
                response = self.session.get(url, timeout=self.timeout_seconds)
            except requests.RequestException as exc:
                if attempt == 4:
                    raise SecError(f"SEC request failed: {url}") from exc
                time.sleep(2 ** (attempt - 1))
                continue
            if response.status_code in {403, 429, 503} and attempt < 4:
                time.sleep(float(response.headers.get("Retry-After", 2**attempt)))
                continue
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise SecError(f"SEC returned HTTP {response.status_code}: {url}") from exc
            return response
        raise SecError(f"SEC request failed: {url}")

    def json(self, url: str) -> dict[str, Any]:
        payload = self.get(url).json()
        if not isinstance(payload, dict):
            raise SecError(f"Expected JSON object: {url}")
        return payload

    def plan(self, tickers: list[str], filings_per_ticker: int) -> list[Filing]:
        if filings_per_ticker <= 0:
            raise ValueError("filings_per_ticker must be positive")
        mapping = {
            str(entry["ticker"]).upper(): entry
            for entry in self.json(TICKER_MAP_URL).values()
            if isinstance(entry, dict) and entry.get("ticker")
        }
        filings: list[Filing] = []
        for raw_ticker in tickers:
            ticker = raw_ticker.upper()
            company = mapping.get(ticker)
            if company is None:
                raise SecError(f"Ticker not found in SEC mapping: {ticker}")
            cik = f"{int(company['cik_str']):010d}"
            payload = self.json(SUBMISSIONS_URL.format(cik=cik))
            candidates = filings_from_submissions(
                ticker, str(company.get("title", ticker)), cik, payload
            )
            if len(candidates) < filings_per_ticker:
                raise SecError(
                    f"Only {len(candidates)} recent 10-K filings found for {ticker}"
                )
            filings.extend(candidates[:filings_per_ticker])
        return filings

    def download_corpus(
        self,
        filings: list[Filing],
        *,
        raw_dir: Path,
        normalized_dir: Path,
        manifest_dir: Path,
        min_item_characters: int = 4_000,
    ) -> dict[str, Any]:
        normalized_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, Any]] = []
        for filing in filings:
            try:
                filing_dir = raw_dir / filing.ticker / filing.report_date
                filing_dir.mkdir(parents=True, exist_ok=True)
                raw_path = filing_dir / filing.primary_document
                content = (
                    raw_path.read_bytes()
                    if raw_path.exists()
                    else self.get(filing.source_url).content
                )
                if not raw_path.exists():
                    raw_path.write_bytes(content)
                item_text = extract_item_7(content, min_characters=min_item_characters)
                output = normalized_dir / (
                    f"{filing.ticker}_{filing.year}_{filing.accession_number}.json"
                )
                output.write_text(
                    json.dumps(normalized_record(filing, item_text, content), indent=2),
                    encoding="utf-8",
                )
                entries.append({**asdict(filing), "status": "ok", "output": str(output)})
            except (OSError, ValueError, requests.RequestException, SecError) as exc:
                entries.append({**asdict(filing), "status": "failed", "error": str(exc)})
        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "requested": len(filings),
            "succeeded": sum(entry["status"] == "ok" for entry in entries),
            "failed": sum(entry["status"] == "failed" for entry in entries),
            "entries": entries,
        }
        manifest_path = manifest_dir / "sec_download.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {**manifest, "manifest": str(manifest_path)}


def filings_from_submissions(
    ticker: str,
    company_name: str,
    cik: str,
    payload: dict[str, Any],
) -> list[Filing]:
    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filings: list[Filing] = []
    for index, form in enumerate(forms):
        if form != "10-K":
            continue
        try:
            filing = Filing(
                ticker=ticker,
                company_name=company_name,
                cik=f"{int(cik):010d}",
                accession_number=str(recent["accessionNumber"][index]),
                filing_date=str(recent["filingDate"][index]),
                report_date=str(recent["reportDate"][index]),
                primary_document=str(recent["primaryDocument"][index]),
            )
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise SecError(f"Incomplete filing metadata for {ticker}") from exc
        filings.append(filing)
    return sorted(filings, key=lambda item: (item.report_date, item.filing_date), reverse=True)


def html_to_text(html: str | bytes) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    for tag in list(soup.find_all(True)):
        if tag.attrs is None:
            continue
        style = str(tag.get("style", "")).replace(" ", "").lower()
        if (
            tag.has_attr("hidden")
            or str(tag.name).lower().endswith(":hidden")
            or "display:none" in style
            or "visibility:hidden" in style
        ):
            tag.decompose()
    lines = [re.sub(r"[\t\f\v\u00a0 ]+", " ", line).strip() for line in soup.get_text("\n").splitlines()]
    return "\n".join(line for line in lines if line)


def extract_item_7(html: str | bytes, *, min_characters: int = 4_000) -> str:
    text = html_to_text(html)
    starts = list(ITEM_7_START.finditer(text))
    ends = list(ITEM_7_END.finditer(text))
    candidates: list[tuple[int, str]] = []
    for start in starts:
        end = next((candidate for candidate in ends if candidate.start() > start.end()), None)
        if end is None:
            continue
        section = "\n".join(
            line.strip() for line in text[start.start() : end.start()].splitlines() if line.strip()
        )
        if len(section) < min_characters:
            continue
        score = len(section) + (100_000 if MDA_TITLE.search(section[:500]) else 0)
        candidates.append((score, section))
    if not candidates:
        raise SecError("Could not locate a sufficiently complete Item 7 section")
    return max(candidates, key=lambda item: item[0])[1]


def normalized_record(filing: Filing, text: str, raw_html: bytes) -> dict[str, Any]:
    return {
        "metadata": {
            "ticker": filing.ticker,
            "company_name": filing.company_name,
            "cik": filing.cik,
            "accession_number": filing.accession_number,
            "filing_type": "10-K",
            "year": filing.year,
            "report_date": filing.report_date,
            "filing_date": filing.filing_date,
            "item_number": "Item 7",
            "item_name": "Management's Discussion and Analysis",
            "provider": "SEC EDGAR",
            "source_url": filing.source_url,
            "raw_sha256": hashlib.sha256(raw_html).hexdigest(),
        },
        "text": text,
    }
