"""
AeroLex — FAA AD (Airworthiness Directive) Ingestor

Downloads FAA Airworthiness Directives from the Federal Register API.
ADs are legally enforceable rules — airlines MUST comply or ground aircraft.

API: https://federalregister.gov/api/v1/documents.json
- Free, no authentication required
- Returns clean JSON with metadata + PDF links
- Updated daily — new ADs issued frequently

What gets downloaded:
- AD metadata (document number, title, abstract, dates)
- PDF content for each AD
- Aircraft type tags for metadata filtering

Why ADs matter for AeroLex:
- United Airlines / Air India must track ALL active ADs for their fleet
- Non-compliance = aircraft grounded by FAA/DGCA
- AeroLex answers: "What are all active ADs for Boeing 787-8?"

Usage:
    from src.ingestion.faa_ad_ingestor import FAAAdIngestor
    ingestor = FAAAdIngestor()
    ingestor.run(max_pages=2)
"""

import os
import json
import time
import hashlib
import requests
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, IngestionError

logger = get_logger(__name__)

# ── API Constants ─────────────────────────────────────────────────────────────
FEDERAL_REGISTER_BASE = "https://federalregister.gov/api/v1"
AD_ENDPOINT           = f"{FEDERAL_REGISTER_BASE}/documents.json"
PER_PAGE              = 20   # Max allowed by API

# ── Aircraft types we care about (United Airlines + Air India fleet) ──────────
TARGET_AIRCRAFT = [
    "boeing 737", "boeing 747", "boeing 757", "boeing 767",
    "boeing 777", "boeing 787",
    "airbus a319", "airbus a320", "airbus a321",
    "airbus a330", "airbus a350", "airbus a380",
]

# ── Data directory ────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path("data/raw/faa_ads")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
HASH_FILE    = RAW_DATA_DIR / "ad_hashes.json"
AD_INDEX     = RAW_DATA_DIR / "ad_index.json"  # Master index of all ADs


class FAAAdIngestor:
    """
    Downloads and manages FAA Airworthiness Directive data.

    Two-phase approach:
    Phase A — Fetch AD metadata list from Federal Register API
    Phase B — Download PDF for each AD (optional, on-demand)

    Why two phases?
    - Metadata fetch is fast (JSON API)
    - PDF download is slow + large files
    - For search/retrieval, metadata + abstract is often enough
    - Full PDF needed only for detailed compliance analysis
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AeroLex-RAG/1.0 (Aviation Compliance Research)",
            "Accept":     "application/json"
        })
        self.hashes   = self._load_hashes()
        self.ad_index = self._load_ad_index()
        self.stats    = {
            "fetched":     0,
            "new":         0,
            "skipped":     0,
            "pdf_downloaded": 0,
            "failed":      0
        }
        logger.info("FAAAdIngestor initialized")

    # ── Hash & Index Management ───────────────────────────────────────────────

    def _load_hashes(self) -> dict:
        if HASH_FILE.exists():
            with open(HASH_FILE) as f:
                return json.load(f)
        return {}

    def _save_hashes(self) -> None:
        with open(HASH_FILE, "w") as f:
            json.dump(self.hashes, f, indent=2)

    def _load_ad_index(self) -> dict:
        """Master index — tracks all ADs we've seen."""
        if AD_INDEX.exists():
            with open(AD_INDEX) as f:
                return json.load(f)
        return {}

    def _save_ad_index(self) -> None:
        with open(AD_INDEX, "w") as f:
            json.dump(self.ad_index, f, indent=2)

    def _compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # ── Aircraft Type Tagging ─────────────────────────────────────────────────

    def _extract_aircraft_tags(self, title: str, abstract: str) -> list[str]:
        """
        Extract aircraft type tags from AD title and abstract.
        Used for metadata filtering in Qdrant.

        Args:
            title: AD title
            abstract: AD abstract text

        Returns:
            list: Matched aircraft types (e.g., ["boeing 787", "airbus a350"])
        """
        combined = (title + " " + abstract).lower()
        tags = []
        for aircraft in TARGET_AIRCRAFT:
            if aircraft in combined:
                tags.append(aircraft)

        # Also check for generic tags
        if "boeing" in combined and not any("boeing" in t for t in tags):
            tags.append("boeing")
        if "airbus" in combined and not any("airbus" in t for t in tags):
            tags.append("airbus")

        return tags if tags else ["general"]

    # ── API Fetching ──────────────────────────────────────────────────────────

    def _build_api_params(self, page: int = 1) -> dict:
        """
        Build Federal Register API query parameters.

        Why these filters?
        - type=RULE → ADs are published as Final Rules
        - agency=FAA → only FAA documents
        - term=airworthiness directive → filter to ADs specifically
        - order=newest → most recent first (most important for compliance)
        """
        return {
            "conditions[type]":             "RULE",
            "conditions[agencies][]":       "federal-aviation-administration",
            "conditions[term]":             "airworthiness directive",
            "per_page":                     PER_PAGE,
            "page":                         page,
            "order":                        "newest",
            "fields[]": [
                "document_number",
                "title",
                "abstract",
                "publication_date",
                "pdf_url",
                "html_url",
                "type",
                "agencies",
                "excerpts"
            ]
        }

    def _fetch_ad_page(self, page: int = 1) -> Optional[dict]:
        """
        Fetch one page of AD results from Federal Register API.

        Args:
            page: Page number (1-indexed)

        Returns:
            dict: API response with results list
        """
        try:
            params = self._build_api_params(page)
            logger.info(f"Fetching AD page {page}...")

            response = self.session.get(AD_ENDPOINT, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            count = len(data.get("results", []))
            total = data.get("count", 0)

            logger.info(
                f"Page {page} fetched | "
                f"Results: {count} | "
                f"Total available: {total}"
            )
            return data

        except Exception as e:
            raise IngestionError(
                message=f"Failed to fetch AD page {page}",
                context="FAAAdIngestor._fetch_ad_page()",
                original_error=e
            )

    def _process_ad_record(self, ad: dict) -> Optional[dict]:
        """
        Process a single AD record from API response.
        Adds aircraft tags, computes hash, checks if new/changed.

        Args:
            ad: Raw AD dict from API

        Returns:
            dict: Processed AD record, or None if unchanged
        """
        doc_number = ad.get("document_number", "unknown")

        # Build structured record
        record = {
            "document_number":  doc_number,
            "title":            ad.get("title", ""),
            "abstract":         ad.get("abstract", ""),
            "publication_date": ad.get("publication_date", ""),
            "pdf_url":          ad.get("pdf_url", ""),
            "html_url":         ad.get("html_url", ""),
            "ingestion_date":   date.today().isoformat(),
            "source":           "federal_register",
            "doc_type":         "airworthiness_directive",
            "aircraft_tags":    self._extract_aircraft_tags(
                                    ad.get("title", ""),
                                    ad.get("abstract", "")
                                ),
            "pdf_downloaded":   False,
            "pdf_path":         None
        }

        # Hash check — skip if unchanged
        content_str = json.dumps({
            "title": record["title"],
            "abstract": record["abstract"],
            "publication_date": record["publication_date"]
        }, sort_keys=True)
        new_hash = self._compute_hash(content_str)

        if self.hashes.get(doc_number) == new_hash:
            self.stats["skipped"] += 1
            return None

        # Update hash
        self.hashes[doc_number] = new_hash
        self.stats["new"] += 1

        return record

    def _download_ad_pdf(self, record: dict) -> bool:
        """
        Download PDF for a specific AD.

        Args:
            record: Processed AD record with pdf_url

        Returns:
            bool: True if downloaded successfully
        """
        pdf_url = record.get("pdf_url")
        doc_number = record.get("document_number", "unknown")

        if not pdf_url:
            logger.warning(f"No PDF URL for AD {doc_number}")
            return False

        # Sanitize filename
        safe_name = doc_number.replace("/", "-").replace(" ", "_")
        pdf_path  = RAW_DATA_DIR / f"ad_{safe_name}.pdf"

        # Skip if already downloaded
        if pdf_path.exists():
            logger.debug(f"PDF already exists: {pdf_path.name}")
            record["pdf_downloaded"] = True
            record["pdf_path"]       = str(pdf_path)
            return True

        try:
            logger.info(f"Downloading PDF for AD {doc_number}...")
            response = self.session.get(pdf_url, timeout=60)
            response.raise_for_status()

            with open(pdf_path, "wb") as f:
                f.write(response.content)

            record["pdf_downloaded"] = True
            record["pdf_path"]       = str(pdf_path)
            self.stats["pdf_downloaded"] += 1

            logger.info(f"PDF downloaded: {pdf_path.name} ({len(response.content)//1024}KB)")
            return True

        except Exception as e:
            logger.warning(f"PDF download failed for {doc_number}: {e}")
            return False

    # ── Main Ingestion ────────────────────────────────────────────────────────

    def run(
        self,
        max_pages: int = 3,
        download_pdfs: bool = False,
        aircraft_filter: Optional[str] = None
    ) -> dict:
        """
        Run full AD ingestion.

        Args:
            max_pages: Number of API pages to fetch (20 ADs per page)
            download_pdfs: If True, download PDF for each AD
            aircraft_filter: Optional aircraft type to filter
                             (e.g., "boeing 787")

        Returns:
            dict: Ingestion stats
        """
        logger.info(
            f"Starting FAA AD ingestion | "
            f"Max pages: {max_pages} | "
            f"Download PDFs: {download_pdfs}"
        )

        for page in range(1, max_pages + 1):
            try:
                data = self._fetch_ad_page(page)
                results = data.get("results", [])

                if not results:
                    logger.info(f"No more results at page {page}")
                    break

                for ad in results:
                    self.stats["fetched"] += 1
                    record = self._process_ad_record(ad)

                    if record is None:
                        continue  # Unchanged — skip

                    # Apply aircraft filter if specified
                    if aircraft_filter:
                        tags = record.get("aircraft_tags", [])
                        if not any(aircraft_filter.lower() in t for t in tags):
                            logger.debug(
                                f"AD {record['document_number']} "
                                f"doesn't match filter '{aircraft_filter}' — skipping"
                            )
                            continue

                    # Download PDF if requested
                    if download_pdfs and record.get("pdf_url"):
                        self._download_ad_pdf(record)
                        time.sleep(0.5)  # Rate limiting for PDFs

                    # Save to index
                    self.ad_index[record["document_number"]] = record

                # Save after each page
                self._save_hashes()
                self._save_ad_index()

                logger.info(
                    f"Page {page} complete | "
                    f"New: {self.stats['new']} | "
                    f"Skipped: {self.stats['skipped']}"
                )

                # Rate limiting between pages
                time.sleep(0.5)

            except IngestionError as e:
                handle_exception(e, context=f"FAAAdIngestor.run() page {page}")
                self.stats["failed"] += 1
                continue

        # Final summary
        logger.info(
            f"FAA AD ingestion complete | "
            f"Fetched: {self.stats['fetched']} | "
            f"New: {self.stats['new']} | "
            f"Skipped: {self.stats['skipped']} | "
            f"PDFs: {self.stats['pdf_downloaded']} | "
            f"Failed: {self.stats['failed']}"
        )
        return self.stats

    def get_ads_by_aircraft(self, aircraft_type: str) -> list[dict]:
        """
        Get all indexed ADs for a specific aircraft type.
        Used by the /ad-check API endpoint.

        Args:
            aircraft_type: e.g., "boeing 787"

        Returns:
            list: Matching AD records
        """
        matches = []
        for doc_number, record in self.ad_index.items():
            tags = record.get("aircraft_tags", [])
            if any(aircraft_type.lower() in t for t in tags):
                matches.append(record)

        logger.info(
            f"Found {len(matches)} ADs for aircraft: {aircraft_type}"
        )
        return matches


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Testing FAA AD Ingestor ---\n")
    print("⚠️  Fetching 2 pages (40 ADs) from Federal Register API\n")

    ingestor = FAAAdIngestor()

    # Fetch 2 pages, no PDF download (fast test)
    stats = ingestor.run(max_pages=2, download_pdfs=False)

    print(f"\n📊 Ingestion Stats:")
    print(f"   Fetched  : {stats['fetched']}")
    print(f"   New      : {stats['new']}")
    print(f"   Skipped  : {stats['skipped']}")
    print(f"   Failed   : {stats['failed']}")

    # Test aircraft filter
    print(f"\n🔍 Boeing 787 ADs found:")
    b787_ads = ingestor.get_ads_by_aircraft("boeing 787")
    for ad in b787_ads[:3]:
        print(f"   [{ad['document_number']}] {ad['title'][:80]}...")
        print(f"   Tags: {ad['aircraft_tags']}")
        print(f"   Date: {ad['publication_date']}")
        print()

    print(f"Total ADs indexed: {len(ingestor.ad_index)}")
    print(f"\n✅ FAA AD Ingestor working!")