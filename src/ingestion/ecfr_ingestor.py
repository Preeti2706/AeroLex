"""
AeroLex — eCFR (Electronic Code of Federal Regulations) Ingestor

Downloads Title 14 (Aeronautics and Space) regulations from the
eCFR REST API and saves them locally with hash-based change detection.

API used: https://ecfr.gov/api/versioner/v1/
- Free, no auth required
- Updated daily by the Government Publishing Office (GPO)

What this ingestor does:
1. Fetches Title 14 structure (chapters → subchapters → parts)
2. Downloads full text for each relevant Part
3. Computes hash of each Part — skips if unchanged since last run
4. Saves raw XML/JSON to data/raw/ecfr/
5. Stores metadata in SQLite (part number, date, hash, file path)

Why hash-based change detection?
- eCFR updates daily but most parts don't change every day
- Without hashing: re-download + re-embed entire corpus daily = expensive
- With hashing: only changed parts get re-processed = efficient + cheap

Usage:
    from src.ingestion.ecfr_ingestor import ECFRIngestor
    ingestor = ECFRIngestor()
    ingestor.run()
"""

import os
import json
import time
import hashlib
import requests
from datetime import datetime, date
from pathlib import Path
from typing import Optional
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, IngestionError
from config.settings import settings
from datetime import date, timedelta

logger = get_logger(__name__)

# ── eCFR API Constants ────────────────────────────────────────────────────────
ECFR_BASE_URL       = "https://ecfr.gov/api/versioner/v1"
TITLE_NUMBER        = 14          # Title 14 = Aeronautics and Space

TODAY = (date.today() - timedelta(days=2)).isoformat()

# ── Parts we care about for AeroLex ──────────────────────────────────────────
# Why these parts? Directly relevant to airline compliance queries
TARGET_PARTS = {
    "1":   "Definitions and Abbreviations",
    "5":   "Safety Management Systems",
    "21":  "Certification Procedures for Products and Articles",
    "25":  "Airworthiness Standards: Transport Category Airplanes",
    "39":  "Airworthiness Directives",
    "43":  "Maintenance, Preventive Maintenance, Rebuilding and Alteration",
    "61":  "Certification: Pilots, Flight Instructors, and Ground Instructors",
    "91":  "General Operating and Flight Rules",
    "117": "Flight and Duty Limitations and Rest Requirements",
    "119": "Certification: Air Carriers and Commercial Operators",
    "121": "Operating Requirements: Domestic, Flag, and Supplemental Operations",
    "135": "Operating Requirements: Commuter and On Demand Operations",
    "145": "Repair Stations",
    "183": "Representatives of the Administrator",
}

# ── Data directory ────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path("data/raw/ecfr")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Hash file — tracks last known hash of each part
HASH_FILE = RAW_DATA_DIR / "part_hashes.json"


class ECFRIngestor:
    """
    Downloads and manages eCFR Title 14 regulatory data.

    Design decisions:
    - Uses requests (not async) — simple, reliable for batch downloads
    - Hash-based change detection — avoids redundant processing
    - Saves raw JSON — preserves original structure for parser
    - Rate limiting (1 sec between calls) — be polite to government API
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AeroLex-RAG/1.0 (Aviation Compliance Research Tool)",
            "Accept": "application/json"
        })
        self.hashes = self._load_hashes()
        self.stats = {
            "downloaded": 0,
            "skipped_unchanged": 0,
            "failed": 0,
            "total_parts": len(TARGET_PARTS)
        }
        logger.info(f"ECFRIngestor initialized | Target parts: {len(TARGET_PARTS)}")

    # ── Hash Management ───────────────────────────────────────────────────────

    def _load_hashes(self) -> dict:
        """Load previously saved hashes from disk."""
        if HASH_FILE.exists():
            with open(HASH_FILE, "r") as f:
                hashes = json.load(f)
            logger.info(f"Loaded {len(hashes)} existing part hashes")
            return hashes
        logger.info("No existing hashes found — fresh ingestion")
        return {}

    def _save_hashes(self) -> None:
        """Save current hashes to disk."""
        with open(HASH_FILE, "w") as f:
            json.dump(self.hashes, f, indent=2)
        logger.debug(f"Saved {len(self.hashes)} hashes to {HASH_FILE}")

    def _compute_hash(self, content: str) -> str:
        """Compute SHA256 hash of content string."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _is_changed(self, part_number: str, new_hash: str) -> bool:
        """
        Check if a part has changed since last ingestion.

        Returns:
            True if changed (needs re-processing)
            False if unchanged (skip)
        """
        old_hash = self.hashes.get(f"part_{part_number}")
        if old_hash is None:
            logger.debug(f"Part {part_number}: New — no previous hash")
            return True
        if old_hash != new_hash:
            logger.info(f"Part {part_number}: Changed — hash mismatch")
            return True
        logger.debug(f"Part {part_number}: Unchanged — skipping")
        return False

    # ── API Calls ─────────────────────────────────────────────────────────────

    def _get_title_structure(self) -> Optional[dict]:
        """
        Fetch Title 14 full structure from eCFR API.

        Returns:
            dict: Title structure with chapters/parts hierarchy
        """
        url = f"{ECFR_BASE_URL}/structure/{TODAY}/title-{TITLE_NUMBER}.json"
        logger.info(f"Fetching Title {TITLE_NUMBER} structure from: {url}")

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Title {TITLE_NUMBER} structure fetched successfully")
            return data

        except requests.exceptions.HTTPError as e:
            # Try yesterday's date if today's not available yet
            logger.warning(f"Today's structure not available — trying without date")
            try:
                url_fallback = f"{ECFR_BASE_URL}/structure/current/title-{TITLE_NUMBER}.json"
                response = self.session.get(url_fallback, timeout=30)
                response.raise_for_status()
                return response.json()
            except Exception as e2:
                raise IngestionError(
                    message=f"Failed to fetch Title 14 structure",
                    context="ECFRIngestor._get_title_structure()",
                    original_error=e2
                )

        except Exception as e:
            raise IngestionError(
                message="Failed to fetch Title 14 structure",
                context="ECFRIngestor._get_title_structure()",
                original_error=e
            )

    def _get_part_content(self, part_number: str) -> Optional[dict]:

        """
        Fetch full text content of a specific Part.

        Args:
            part_number: CFR Part number (e.g., "121")

        Returns:
            dict: Full part content as parsed XML
        """

        url = (
            f"{ECFR_BASE_URL}/full/{TODAY}/title-{TITLE_NUMBER}.xml"
            f"?part={part_number}"
        )

        try:
            logger.info(f"Fetching Part {part_number} content...")
            response = self.session.get(url, timeout=60)
            response.raise_for_status()

            # API returns XML — parse it to dict using lxml
            from lxml import etree
            root = etree.fromstring(response.content)

            # Convert to dict for consistent handling
            content = {
                "part_number": part_number,
                "fetched_date": TODAY,
                "raw_xml": response.text,
                "encoding": response.encoding or "utf-8"
            }
            return content

        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                logger.warning(f"Part {part_number} not found in eCFR — skipping")
                return None
            raise IngestionError(
                message=f"HTTP error fetching Part {part_number}: {response.status_code}",
                context="ECFRIngestor._get_part_content()",
                original_error=e
            )

        except Exception as e:
            raise IngestionError(
                message=f"Failed to fetch Part {part_number}",
                context="ECFRIngestor._get_part_content()",
                original_error=e
            )

    # ── Save to Disk ──────────────────────────────────────────────────────────

    def _save_part(self, part_number: str, content: dict) -> Path:
        """
            Save part content to disk — raw XML + metadata JSON.

            Args:
                part_number: CFR Part number
                content: Part content dict with raw_xml key

            Returns:
                Path: Path where XML file was saved
        """
        # Save raw XML
        xml_path = RAW_DATA_DIR / f"part_{part_number}_{TODAY}.xml"
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(content["raw_xml"])

        # Save metadata JSON
        meta_path = RAW_DATA_DIR / f"part_{part_number}_{TODAY}_meta.json"
        meta = {
            "part_number": part_number,
            "part_name": TARGET_PARTS.get(part_number, "Unknown"),
            "fetched_date": content["fetched_date"],
            "file_path": str(xml_path),
            "source": "ecfr_api",
            "title": TITLE_NUMBER
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Part {part_number} saved to: {xml_path}")
        return xml_path

    # ── Main Ingestion Logic ──────────────────────────────────────────────────

    def ingest_part(self, part_number: str, part_name: str) -> bool:
        """
        Ingest a single CFR Part with hash-based change detection.

        Args:
            part_number: CFR Part number (e.g., "121")
            part_name: Human-readable part name

        Returns:
            bool: True if downloaded, False if skipped/failed
        """
        try:
            # Fetch content
            content = self._get_part_content(part_number)
            if content is None:
                self.stats["failed"] += 1
                return False

            # Compute hash
            content_str = json.dumps(content, sort_keys=True)
            new_hash = self._compute_hash(content_str)

            # Check if changed
            if not self._is_changed(part_number, new_hash):
                self.stats["skipped_unchanged"] += 1
                return False

            # Save to disk
            file_path = self._save_part(part_number, content)

            # Update hash
            self.hashes[f"part_{part_number}"] = new_hash
            self._save_hashes()

            self.stats["downloaded"] += 1
            logger.info(
                f"✅ Part {part_number} ({part_name}) ingested | "
                f"File: {file_path.name}"
            )
            return True

        except IngestionError as e:
            handle_exception(e, context=f"ECFRIngestor.ingest_part({part_number})")
            self.stats["failed"] += 1
            return False

        except Exception as e:
            handle_exception(e, context=f"ECFRIngestor.ingest_part({part_number})")
            self.stats["failed"] += 1
            return False

    def run(self, parts: Optional[list] = None) -> dict:
        """
        Run full ingestion for all target parts.

        Args:
            parts: Optional list of specific part numbers to ingest.
                   If None, ingests all TARGET_PARTS.

        Returns:
            dict: Ingestion stats
        """
        target = parts or list(TARGET_PARTS.keys())
        logger.info(f"Starting eCFR ingestion | Parts to process: {len(target)}")

        for part_number in target:
            part_name = TARGET_PARTS.get(part_number, "Unknown")
            logger.info(f"Processing Part {part_number}: {part_name}")

            self.ingest_part(part_number, part_name)

            # Rate limiting — be polite to government API
            # 1 second between calls = ~14 seconds for all parts
            time.sleep(1)

        # Final summary
        logger.info(
            f"eCFR ingestion complete | "
            f"Downloaded: {self.stats['downloaded']} | "
            f"Skipped (unchanged): {self.stats['skipped_unchanged']} | "
            f"Failed: {self.stats['failed']} | "
            f"Total: {self.stats['total_parts']}"
        )
        return self.stats


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Testing eCFR Ingestor ---\n")
    print("⚠️  This will make real API calls to ecfr.gov")
    print("⚠️  Downloading Part 1 and Part 91 only for test\n")

    ingestor = ECFRIngestor()

    # Test with just 2 parts first
    stats = ingestor.run(parts=["1", "91"])

    print(f"\n📊 Ingestion Stats:")
    print(f"   Downloaded : {stats['downloaded']}")
    print(f"   Skipped    : {stats['skipped_unchanged']}")
    print(f"   Failed     : {stats['failed']}")
    print(f"\nCheck data/raw/ecfr/ for downloaded files")