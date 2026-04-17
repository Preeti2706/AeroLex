"""
AeroLex — DGCA (Directorate General of Civil Aviation) CAR Ingestor

Scrapes Civil Aviation Requirements (CARs) from dgca.gov.in using
Playwright browser automation.

Why Playwright instead of requests?
- DGCA portal is JavaScript-rendered (React/Angular based)
- Simple HTTP requests return empty page — no content
- Playwright launches real browser, waits for JS to execute
- Then extracts fully rendered HTML + PDF links

What gets downloaded:
- CAR metadata (section, part, subject, issue date, amendment date)
- PDF files for each CAR
- Stored with hash-based change detection

Target Sections (most relevant for airline compliance):
- Section 2: Airworthiness
- Section 3: Air Transport (AOC requirements)
- Section 5: Air Safety
- Section 7: Flight Crew Standards

PDF URL Discovery (Key Insight):
- DGCA renders PDF links as <a data-url="dynamicPdf/..."> in nav area
- Multiple "Part I" exist across series — dict overwrites duplicates
- Solution: collect PDF URLs as ORDERED LIST, assign positionally
- PDF links and non-revoked table rows appear in same order on page

Usage:
    from src.ingestion.dgca_ingestor import DGCAIngestor
    ingestor = DGCAIngestor()
    ingestor.run()
"""

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

# ── Constants — loaded from settings, not hardcoded ──────────────────────────
DGCA_BASE_URL = settings.DGCA_BASE_URL

TARGET_SECTIONS = {
    "2": {"name": "Airworthiness",         "id": settings.DGCA_SECTION_2_ID},
    "3": {"name": "Air Transport",         "id": settings.DGCA_SECTION_3_ID},
    "5": {"name": "Air Safety",            "id": settings.DGCA_SECTION_5_ID},
    "7": {"name": "Flight Crew Standards", "id": settings.DGCA_SECTION_7_ID},
}

# ── Data directory ────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path("data/raw/dgca")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
HASH_FILE    = RAW_DATA_DIR / "car_hashes.json"
CAR_INDEX    = RAW_DATA_DIR / "car_index.json"


class DGCAIngestor:
    """
    Scrapes DGCA Civil Aviation Requirements using Playwright.

    Two-phase approach:
    Phase A — Scrape CAR listing pages (metadata + PDF URLs)
    Phase B — Download PDFs for each CAR

    Key design decision — ordered list for PDF matching:
    - DGCA has duplicate part names ("Part I") across series
    - Using dict loses duplicates — wrong PDFs get assigned
    - Ordered list + positional assignment = correct matching
    """

    def __init__(self, headless: bool = True):
        """
        Args:
            headless: Run browser in headless mode (no GUI)
                      Set False for debugging to see browser
        """
        self.headless  = headless
        self.hashes    = self._load_hashes()
        self.car_index = self._load_car_index()
        self.stats     = {
            "sections_scraped": 0,
            "cars_found":       0,
            "new":              0,
            "skipped":          0,
            "pdfs_downloaded":  0,
            "failed":           0
        }
        self.http = requests.Session()
        self.http.headers.update({
            "User-Agent": "AeroLex-RAG/1.0 (Aviation Compliance Research)"
        })
        logger.info(f"DGCAIngestor initialized | Headless: {headless}")

    # ── Hash & Index Management ───────────────────────────────────────────────

    def _load_hashes(self) -> dict:
        if HASH_FILE.exists():
            with open(HASH_FILE) as f:
                return json.load(f)
        return {}

    def _save_hashes(self) -> None:
        with open(HASH_FILE, "w") as f:
            json.dump(self.hashes, f, indent=2)

    def _load_car_index(self) -> dict:
        if CAR_INDEX.exists():
            with open(CAR_INDEX) as f:
                return json.load(f)
        return {}

    def _save_car_index(self) -> None:
        with open(CAR_INDEX, "w") as f:
            json.dump(self.car_index, f, indent=2)

    def _compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # ── PDF Link Collection ───────────────────────────────────────────────────

    def _collect_pdf_links_ordered(self, page) -> list[str]:
        """
        Collect ALL dynamicPdf links as ORDERED LIST.

        Why ordered list instead of dict?
        - Multiple "Part I" exist across different series
        - Dict would overwrite duplicates — wrong URLs assigned
        - Ordered list preserves exact page order
        - PDF links and valid (non-revoked) table rows
          appear in same order on page — positional matching works

        Args:
            page: Playwright page object

        Returns:
            list: PDF URLs in page order
        """
        pdf_urls = []
        try:
            all_pdf_links = page.query_selector_all("a[data-url*='dynamicPdf']")
            logger.info(f"Found {len(all_pdf_links)} dynamicPdf links on page")

            for link in all_pdf_links:
                data_url = link.get_attribute("data-url")
                if data_url:
                    full_url = (
                        f"{DGCA_BASE_URL}?baseLocale=en_US?"
                        f"dynamicPage={data_url}"
                    )
                    pdf_urls.append(full_url)
                    logger.debug(f"PDF URL collected: {data_url[:50]}")

        except Exception as e:
            logger.warning(f"PDF link collection error: {e}")

        return pdf_urls

    # ── Playwright Scraping ───────────────────────────────────────────────────

    def _scrape_section(self, section_num: str, section_info: dict) -> list[dict]:
        """
        Scrape one DGCA CAR section using Playwright.

        Steps:
        1. Navigate to section page
        2. Wait for JS to render table
        3. Collect ALL PDF links as ordered list
        4. Parse table rows — skip revoked, assign PDF positionally
        5. Return list of CAR records

        Args:
            section_num: Section number (e.g., "3")
            section_info: Dict with name and portal ID

        Returns:
            list: CAR records found in this section
        """
        from playwright.sync_api import sync_playwright

        section_url = (
            f"{DGCA_BASE_URL}?baseLocale=en_US?"
            f"dynamicPage=CivilAviationReqContent/6/{section_info['id']}/"
            f"viewDynamicRuleContLvl2/html&main"
            f"civilAviationRequirements/6/0/viewDynamicRulesReq"
        )

        logger.info(
            f"Scraping Section {section_num}: {section_info['name']} | "
            f"URL: {section_url}"
        )

        cars = []

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.headless)
                page    = browser.new_page()

                # Navigate and wait for full JS render
                page.goto(section_url, wait_until="networkidle", timeout=30000)

                # Wait for table
                try:
                    page.wait_for_selector("table", timeout=15000)
                    logger.info(f"Section {section_num}: Table loaded")
                except Exception:
                    logger.warning(f"Section {section_num}: No table found")
                    browser.close()
                    return []

                # ── Step 1: Collect PDF URLs as ordered list ──────────────
                # MUST do before iterating rows
                # PDF links appear in nav area, not inside table rows
                pdf_urls  = self._collect_pdf_links_ordered(page)
                pdf_index = 0  # Positional pointer into pdf_urls

                # ── Step 2: Parse table rows ──────────────────────────────
                rows = page.query_selector_all("table tr")
                logger.info(f"Section {section_num}: Found {len(rows)} rows")

                current_series = ""

                for row in rows:
                    cells = row.query_selector_all("td")

                    # Series header row (e.g., "SERIES C - AIR OPERATORS...")
                    if len(cells) == 1:
                        text = cells[0].inner_text().strip()
                        if "SERIES" in text.upper():
                            current_series = text
                        continue

                    # CAR data row — expect 3+ columns
                    if len(cells) < 3:
                        continue

                    try:
                        part         = cells[0].inner_text().strip()
                        issue_info   = cells[1].inner_text().strip()
                        subject      = cells[2].inner_text().strip() if len(cells) > 2 else ""
                        amendment_no = cells[3].inner_text().strip() if len(cells) > 3 else ""
                        amend_date   = cells[4].inner_text().strip() if len(cells) > 4 else ""

                        # Skip revoked CARs — no PDF assigned for these
                        if "REVOKED" in issue_info.upper() or "REVOKED" in subject.upper():
                            logger.debug(f"Skipping revoked: Section {section_num} {part}")
                            continue

                        # Skip header rows
                        if "CAR SERIES" in part.upper() or "ISSUE NO" in issue_info.upper():
                            continue

                        # Skip empty rows
                        if not subject or not part:
                            continue

                        # ── Step 3: Assign next PDF URL positionally ───────
                        # Revoked CARs are skipped (no PDF link for them)
                        # Valid CARs match 1:1 with pdf_urls in order
                        pdf_url = None
                        if pdf_index < len(pdf_urls):
                            pdf_url    = pdf_urls[pdf_index]
                            pdf_index += 1
                            logger.debug(
                                f"PDF assigned [{pdf_index}]: {part} → "
                                f"{pdf_url[60:100]}..."
                            )
                        else:
                            logger.debug(f"No PDF available for: {part}")

                        car_id = f"CAR_S{section_num}_{part.replace(' ', '_')}"

                        record = {
                            "car_id":           car_id,
                            "section_number":   section_num,
                            "section_name":     section_info["name"],
                            "series":           current_series,
                            "part":             part,
                            "issue_info":       issue_info,
                            "subject":          subject,
                            "amendment_no":     amendment_no,
                            "amendment_date":   amend_date,
                            "pdf_url":          pdf_url,
                            "ingestion_date":   date.today().isoformat(),
                            "source":           "dgca",
                            "doc_type":         "car",
                            "pdf_downloaded":   False,
                            "pdf_path":         None
                        }
                        cars.append(record)
                        logger.debug(
                            f"Found CAR: Section {section_num} {part} — "
                            f"{subject[:50]}"
                        )

                    except Exception as e:
                        logger.debug(f"Row parsing error: {e}")
                        continue

                browser.close()

        except Exception as e:
            raise IngestionError(
                message=f"Failed to scrape Section {section_num}",
                context="DGCAIngestor._scrape_section()",
                original_error=e
            )

        pdf_count = sum(1 for c in cars if c["pdf_url"])
        logger.info(
            f"Section {section_num} scraped | "
            f"CARs found: {len(cars)} | "
            f"PDFs matched: {pdf_count}/{len(cars)}"
        )
        return cars

    # ── PDF Download ──────────────────────────────────────────────────────────

    def _download_car_pdf(self, record: dict) -> bool:
        """
        Download PDF for a specific CAR.

        Args:
            record: CAR record with pdf_url

        Returns:
            bool: True if downloaded successfully
        """
        pdf_url = record.get("pdf_url")
        car_id  = record.get("car_id", "unknown")

        if not pdf_url:
            logger.debug(f"No PDF URL for CAR {car_id}")
            return False

        safe_name = car_id.replace("/", "-").replace(" ", "_")
        pdf_path  = RAW_DATA_DIR / f"{safe_name}.pdf"

        if pdf_path.exists():
            logger.debug(f"PDF already exists: {pdf_path.name}")
            record["pdf_downloaded"] = True
            record["pdf_path"]       = str(pdf_path)
            return True

        try:
            logger.info(f"Downloading PDF for CAR {car_id}...")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer":    DGCA_BASE_URL,
                "Accept":     "application/pdf,*/*"
            }
            response = self.http.get(pdf_url, headers=headers, timeout=60)
            response.raise_for_status()

            # Verify it's actually a PDF
            if b"%PDF" not in response.content[:10]:
                logger.warning(f"Response is not a PDF for {car_id} — skipping")
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
            logger.warning(f"PDF download failed for {car_id}: {e}")
            return False

    # ── Main Ingestion ────────────────────────────────────────────────────────

    def run(
        self,
        sections: Optional[list] = None,
        download_pdfs: bool = False
    ) -> dict:
        """
        Run full DGCA CAR ingestion.

        Args:
            sections: List of section numbers to scrape.
                      If None, scrapes all TARGET_SECTIONS.
            download_pdfs: If True, download PDF for each CAR.

        Returns:
            dict: Ingestion stats
        """
        target = sections or list(TARGET_SECTIONS.keys())
        logger.info(
            f"Starting DGCA CAR ingestion | "
            f"Sections: {target} | "
            f"Download PDFs: {download_pdfs}"
        )

        for section_num in target:
            if section_num not in TARGET_SECTIONS:
                logger.warning(f"Unknown section: {section_num} — skipping")
                continue

            section_info = TARGET_SECTIONS[section_num]

            try:
                cars = self._scrape_section(section_num, section_info)
                self.stats["sections_scraped"] += 1
                self.stats["cars_found"] += len(cars)

                for car in cars:
                    car_id  = car["car_id"]
                    content = json.dumps({
                        "subject":        car["subject"],
                        "amendment_date": car["amendment_date"],
                        "issue_info":     car["issue_info"]
                    }, sort_keys=True)
                    new_hash = self._compute_hash(content)

                    # Hash check — skip unchanged content
                    # BUT always update pdf_url in case it was missing before
                    if self.hashes.get(car_id) == new_hash:
                        # Update PDF URL even for skipped records
                        if car_id in self.car_index and car.get("pdf_url"):
                            self.car_index[car_id]["pdf_url"] = car["pdf_url"]
                        self.stats["skipped"] += 1
                        continue

                    self.hashes[car_id] = new_hash
                    self.stats["new"]   += 1

                    if download_pdfs and car.get("pdf_url"):
                        self._download_car_pdf(car)
                        time.sleep(1)

                    self.car_index[car_id] = car

                self._save_hashes()
                self._save_car_index()

                logger.info(
                    f"Section {section_num} complete | "
                    f"New: {self.stats['new']} | "
                    f"Skipped: {self.stats['skipped']}"
                )
                time.sleep(2)

            except IngestionError as e:
                handle_exception(
                    e,
                    context=f"DGCAIngestor.run() section {section_num}"
                )
                self.stats["failed"] += 1
                continue

        logger.info(
            f"DGCA ingestion complete | "
            f"Sections: {self.stats['sections_scraped']} | "
            f"CARs found: {self.stats['cars_found']} | "
            f"New: {self.stats['new']} | "
            f"Skipped: {self.stats['skipped']} | "
            f"PDFs: {self.stats['pdfs_downloaded']} | "
            f"Failed: {self.stats['failed']}"
        )
        return self.stats


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Testing DGCA CAR Ingestor ---\n")
    print("⚠️  Launching headless browser to scrape dgca.gov.in")
    print("⚠️  Testing Section 3 (Air Transport) only\n")

    ingestor = DGCAIngestor(headless=True)

    # Delete existing index to force fresh run
    import shutil
    if CAR_INDEX.exists():
        CAR_INDEX.unlink()
    if HASH_FILE.exists():
        HASH_FILE.unlink()

    stats = ingestor.run(sections=["3"], download_pdfs=False)

    print(f"\n📊 Ingestion Stats:")
    print(f"   Sections scraped : {stats['sections_scraped']}")
    print(f"   CARs found       : {stats['cars_found']}")
    print(f"   New              : {stats['new']}")
    print(f"   Skipped          : {stats['skipped']}")
    print(f"   Failed           : {stats['failed']}")

    print(f"\n📋 CARs indexed:")
    for car_id, car in ingestor.car_index.items():
        print(f"   [{car['section_number']} {car['part']}] {car['subject'][:60]}")
        print(f"   Amendment: {car['amendment_date']} | PDF: {'✅' if car['pdf_url'] else '❌'}")
        print()

    print(f"\n✅ DGCA Ingestor working!")