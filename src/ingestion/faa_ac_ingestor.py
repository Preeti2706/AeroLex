"""
AeroLex — FAA AC (Advisory Circular) Ingestor

Downloads FAA Advisory Circular metadata via CSV bulk export
and optionally downloads individual AC PDFs.

Why CSV approach instead of scraping?
- FAA provides official bulk CSV export — clean, reliable
- One request = all 1500+ ACs metadata
- No scraping, no pagination, no rate limiting issues
- CSV updated regularly by FAA

What gets downloaded:
- Phase A: CSV metadata (AC number, title, status, date, PDF URL)
- Phase B: PDF files on-demand for target ACs

Target AC Series for AeroLex:
- Series 20: Airworthiness (most relevant)
- Series 25: Transport Category Airplanes (Boeing 787, Airbus A350)
- Series 60: Flight Crew (pilot procedures)
- Series 91: General Operating Rules
- Series 120: Air Carrier Operations (United Airlines directly)
- Series 121: Air Carrier Certification

Usage:
    from src.ingestion.faa_ac_ingestor import FAAACIngestor
    ingestor = FAAACIngestor()
    ingestor.run()
"""

import csv
import io
import json
import time
import hashlib
import requests
from datetime import date
from pathlib import Path
from typing import Optional
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, IngestionError
from config.settings import settings

logger = get_logger(__name__)

# ── API Constants ─────────────────────────────────────────────────────────────
FAA_AC_CSV_URL  = "https://www.faa.gov/regulations_policies/advisory_circulars/index.cfm/go/document.exportAll/statusID/2"
FAA_AC_BASE_URL = "https://www.faa.gov"

# ── Target AC Series — most relevant for airline compliance ───────────────────
TARGET_SERIES = {
    "20":  "Airworthiness",
    "25":  "Transport Category Airplanes",
    "60":  "Flight Crew Procedures",
    "91":  "General Operating Rules",
    "120": "Air Carrier Operations",
    "121": "Air Carrier Certification",
    "135": "Charter Operations",
    "145": "Repair Stations",
    "150": "Airport Design",
}

# ── Data directory ────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path("data/raw/faa_acs")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
HASH_FILE    = RAW_DATA_DIR / "ac_hashes.json"
AC_INDEX     = RAW_DATA_DIR / "ac_index.json"
CSV_CACHE    = RAW_DATA_DIR / "ac_list_cache.csv"


class FAAACIngestor:
    """
    Downloads and manages FAA Advisory Circular data.

    Two-phase approach:
    Phase A — Download CSV → parse metadata → filter target series
    Phase B — Download PDFs for target ACs (on-demand)

    Why filter by series?
    - 1500+ total ACs — too many to embed all
    - Only ~400 ACs in our target series are relevant
    - Series-based filter keeps corpus focused and costs low
    """

    def __init__(self):
        self.session  = requests.Session()
        self.session.headers.update({
            "User-Agent": "AeroLex-RAG/1.0 (Aviation Compliance Research)",
            "Accept":     "text/csv,*/*"
        })
        self.hashes   = self._load_hashes()
        self.ac_index = self._load_ac_index()
        self.stats    = {
            "total_in_csv":      0,
            "target_series":     0,
            "new":               0,
            "skipped":           0,
            "pdfs_downloaded":   0,
            "failed":            0
        }
        logger.info("FAAACIngestor initialized")

    # ── Hash & Index Management ───────────────────────────────────────────────

    def _load_hashes(self) -> dict:
        if HASH_FILE.exists():
            with open(HASH_FILE) as f:
                return json.load(f)
        return {}

    def _save_hashes(self) -> None:
        with open(HASH_FILE, "w") as f:
            json.dump(self.hashes, f, indent=2)

    def _load_ac_index(self) -> dict:
        if AC_INDEX.exists():
            with open(AC_INDEX) as f:
                return json.load(f)
        return {}

    def _save_ac_index(self) -> None:
        with open(AC_INDEX, "w") as f:
            json.dump(self.ac_index, f, indent=2)

    def _compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # ── Series Detection ──────────────────────────────────────────────────────

    def _get_series(self, ac_number: str) -> Optional[str]:
        """
        Extract series number from AC number.

        AC numbering: {series}-{number}
        Examples:
            "20-69"   → series "20"
            "120-109" → series "120"
            "25.1309" → series "25"

        Args:
            ac_number: AC number string

        Returns:
            str: Series number or None if can't determine
        """
        if not ac_number:
            return None

        # Handle formats: "20-69", "120-109", "25.1309"
        for separator in ["-", "."]:
            if separator in ac_number:
                series = ac_number.split(separator)[0].strip()
                return series

        return ac_number.strip()

    def _is_target_series(self, ac_number: str) -> bool:
        """Check if AC is in our target series."""
        series = self._get_series(ac_number)
        return series in TARGET_SERIES

    # ── CSV Download & Parsing ────────────────────────────────────────────────

    def _download_csv(self, force_refresh: bool = False) -> str:
        """
        Download AC list CSV from FAA.

        Uses cached version if available and not forcing refresh.
        Cache is valid for 7 days — ACs change slowly.

        Args:
            force_refresh: Force re-download even if cache exists

        Returns:
            str: CSV content as string
        """
        # Use cache if available and recent
        if not force_refresh and CSV_CACHE.exists():
            cache_age_days = (
                date.today() -
                date.fromtimestamp(CSV_CACHE.stat().st_mtime)
            ).days
            if cache_age_days < 7:
                logger.info(
                    f"Using cached CSV ({cache_age_days} days old) — "
                    f"use force_refresh=True to re-download"
                )
                with open(CSV_CACHE, encoding="utf-8-sig") as f:
                    return f.read()

        # Download fresh CSV
        logger.info(f"Downloading FAA AC CSV from: {FAA_AC_CSV_URL}")
        try:
            response = self.session.get(FAA_AC_CSV_URL, timeout=60)
            response.raise_for_status()

            csv_content = response.text

            # Cache the CSV
            with open(CSV_CACHE, "w", encoding="utf-8") as f:
                f.write(csv_content)

            logger.info(
                f"CSV downloaded and cached | "
                f"Size: {len(csv_content)//1024}KB"
            )
            return csv_content

        except Exception as e:
            raise IngestionError(
                message="Failed to download FAA AC CSV",
                context="FAAACIngestor._download_csv()",
                original_error=e
            )

    def _parse_csv(self, csv_content: str) -> list[dict]:
        """
        Parse AC CSV into list of AC records.

        CSV columns (from FAA export):
        DocumentNumber, Title, Status, SubjectArea,
        OfficeCode, DocumentDate, URL

        Args:
            csv_content: Raw CSV string

        Returns:
            list: Parsed AC records
        """
        records = []
        try:
            # Handle BOM and encoding
            if csv_content.startswith('\ufeff'):
                csv_content = csv_content[1:]

            reader = csv.DictReader(io.StringIO(csv_content))

            for row in reader:
                # Normalize keys — CSV headers may have spaces
                normalized = {k.strip(): v.strip() for k, v in row.items()}

                # Extract fields — handle different possible column names
                ac_number = (
                    normalized.get("DOCUMENTNUMBER") or
                    normalized.get("DocumentNumber") or
                    normalized.get("Number") or ""
                ).strip()

                title = (
                    normalized.get("TITLE") or
                    normalized.get("Title") or ""
                ).strip()

                status = normalized.get("Status", "Active").strip()

                ac_date = (
                    normalized.get("DATE") or
                    normalized.get("DocumentDate") or
                    normalized.get("Date") or ""
                ).strip()

                subject_area = (
                    normalized.get("SubjectArea") or
                    normalized.get("Subject Area") or ""
                ).strip()

                office = (
                    normalized.get("OFFICE") or
                    normalized.get("OfficeCode") or ""
                ).strip()

                # PDF URL — construct from AC number if not in CSV
                pdf_url = normalized.get("URL", "").strip()
                if not pdf_url and ac_number:
                    # FAA PDF URL pattern
                    safe_num = ac_number.replace(" ", "_")
                    pdf_url  = (
                        f"{FAA_AC_BASE_URL}/documentLibrary/media/"
                        f"Advisory_Circular/AC_{safe_num}.pdf"
                    )

                # Skip empty rows
                if not any(normalized.values()):
                    continue
                
                if not ac_number:
                    continue

                record = {
                    "ac_number":      ac_number,
                    "title":          title,
                    "status":         status,
                    "subject_area":   subject_area,
                    "office":         office,
                    "ac_date":        ac_date,
                    "pdf_url":        pdf_url,
                    "series":         self._get_series(ac_number),
                    "ingestion_date": date.today().isoformat(),
                    "source":         "faa_ac",
                    "doc_type":       "advisory_circular",
                    "pdf_downloaded": False,
                    "pdf_path":       None
                }
                records.append(record)

        except Exception as e:
            raise IngestionError(
                message="Failed to parse AC CSV",
                context="FAAACIngestor._parse_csv()",
                original_error=e
            )

        logger.info(f"CSV parsed | Total ACs: {len(records)}")
        return records

    # ── PDF Download ──────────────────────────────────────────────────────────

    def _download_ac_pdf(self, record: dict) -> bool:
        """
        Download PDF for a specific AC.

        Args:
            record: AC record with pdf_url

        Returns:
            bool: True if downloaded successfully
        """
        pdf_url   = record.get("pdf_url")
        ac_number = record.get("ac_number", "unknown")

        if not pdf_url:
            logger.debug(f"No PDF URL for AC {ac_number}")
            return False

        safe_name = ac_number.replace("/", "-").replace(" ", "_").replace(".", "_")
        pdf_path  = RAW_DATA_DIR / f"ac_{safe_name}.pdf"

        if pdf_path.exists():
            logger.debug(f"PDF already exists: {pdf_path.name}")
            record["pdf_downloaded"] = True
            record["pdf_path"]       = str(pdf_path)
            return True

        try:
            logger.info(f"Downloading PDF for AC {ac_number}...")
            response = self.session.get(pdf_url, timeout=60)
            response.raise_for_status()

            if b"%PDF" not in response.content[:10]:
                logger.warning(f"Response is not PDF for AC {ac_number}")
                return False

            with open(pdf_path, "wb") as f:
                f.write(response.content)

            record["pdf_downloaded"] = True
            record["pdf_path"]       = str(pdf_path)
            self.stats["pdfs_downloaded"] += 1

            logger.info(
                f"PDF downloaded: {pdf_path.name} "
                f"({len(response.content)//1024}KB)"
            )
            return True

        except Exception as e:
            logger.warning(f"PDF download failed for AC {ac_number}: {e}")
            return False

    # ── Main Ingestion ────────────────────────────────────────────────────────

    def run(
        self,
        target_series: Optional[list] = None,
        download_pdfs: bool = False,
        force_refresh: bool = False,
        max_pdfs: int = 10
    ) -> dict:
        """
        Run full AC ingestion.

        Args:
            target_series: List of series to ingest.
                           If None, uses TARGET_SERIES.
            download_pdfs: If True, download PDFs.
            force_refresh: Force re-download of CSV.
            max_pdfs: Max PDFs to download (cost control).

        Returns:
            dict: Ingestion stats
        """
        series_filter = target_series or list(TARGET_SERIES.keys())
        logger.info(
            f"Starting FAA AC ingestion | "
            f"Series filter: {series_filter} | "
            f"Download PDFs: {download_pdfs}"
        )

        try:
            # Phase A: Download + parse CSV
            csv_content = self._download_csv(force_refresh=force_refresh)
            all_records = self._parse_csv(csv_content)
            self.stats["total_in_csv"] = len(all_records)

            # Filter to target series
            target_records = [
                r for r in all_records
                if r["series"] in series_filter
            ]
            self.stats["target_series"] = len(target_records)

            logger.info(
                f"Filtered to target series | "
                f"Total: {len(all_records)} | "
                f"Target: {len(target_records)}"
            )

            # Phase B: Process each AC
            pdf_count = 0
            for record in target_records:
                ac_number = record["ac_number"]

                # Hash check
                content_str = json.dumps({
                    "title":   record["title"],
                    "ac_date": record["ac_date"],
                    "status":  record["status"]
                }, sort_keys=True)
                new_hash = self._compute_hash(content_str)

                if self.hashes.get(ac_number) == new_hash:
                    self.stats["skipped"] += 1
                    continue

                self.hashes[ac_number] = new_hash
                self.stats["new"] += 1
                self.ac_index[ac_number] = record

                # Download PDF if requested and under limit
                if download_pdfs and record.get("pdf_url") and pdf_count < max_pdfs:
                    success = self._download_ac_pdf(record)
                    if success:
                        pdf_count += 1
                    time.sleep(0.5)

            # Save
            self._save_hashes()
            self._save_ac_index()

            logger.info(
                f"FAA AC ingestion complete | "
                f"Total in CSV: {self.stats['total_in_csv']} | "
                f"Target series: {self.stats['target_series']} | "
                f"New: {self.stats['new']} | "
                f"Skipped: {self.stats['skipped']} | "
                f"PDFs: {self.stats['pdfs_downloaded']}"
            )

        except IngestionError as e:
            handle_exception(e, context="FAAACIngestor.run()")
            self.stats["failed"] += 1

        return self.stats

    def get_acs_by_series(self, series: str) -> list[dict]:
        """
        Get all indexed ACs for a specific series.

        Args:
            series: Series number e.g. "120"

        Returns:
            list: Matching AC records
        """
        matches = [
            r for r in self.ac_index.values()
            if r.get("series") == series
        ]
        logger.info(f"Found {len(matches)} ACs for series {series}")
        return matches


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Testing FAA AC Ingestor ---\n")
    print("⚠️  Downloading AC CSV from FAA (one-time, cached after)\n")

    ingestor = FAAACIngestor()
    stats    = ingestor.run(
        target_series=["120", "121"],  # Air carrier series only for test
        download_pdfs=False,
        force_refresh=True
    )

    print(f"\n📊 Ingestion Stats:")
    print(f"   Total in CSV     : {stats['total_in_csv']}")
    print(f"   Target series    : {stats['target_series']}")
    print(f"   New              : {stats['new']}")
    print(f"   Skipped          : {stats['skipped']}")
    print(f"   Failed           : {stats['failed']}")

    # Show sample ACs
    print(f"\n📋 Sample Series 120 ACs:")
    s120 = ingestor.get_acs_by_series("120")
    for ac in s120[:5]:
        print(f"   [{ac['ac_number']}] {ac['title'][:60]}")
        print(f"   Date: {ac['ac_date']} | Series: {ac['series']}")
        print()

    print(f"Total ACs indexed: {len(ingestor.ac_index)}")
    print(f"\n✅ FAA AC Ingestor working!")