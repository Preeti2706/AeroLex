"""
AeroLex — SKYbrary Ingestor

Scrapes aviation safety articles from skybrary.aero using
BeautifulSoup (BS4) HTML parsing.

Why BeautifulSoup instead of Playwright?
- SKYbrary is server-side rendered — no JavaScript needed
- Simple requests + BS4 is faster and lighter than Playwright
- No browser overhead for static HTML content

What gets scraped:
- Accidents & Incidents articles
- Safety topic articles (CFIT, ALAR, etc.)
- Article text, categories, tags, related articles

Scraping strategy:
1. Fetch article listing pages (paginated)
2. Extract article URLs
3. Scrape each article for full text + metadata
4. Hash-based change detection — skip unchanged articles

Why SKYbrary for AeroLex?
- ICAO-backed content — authoritative safety knowledge
- Real accident case studies — "why did this happen?"
- ICAO Annex summaries — regulatory context
- Complements FAA regulatory data with safety analysis

Usage:
    from src.ingestion.skybrary_ingestor import SKYbraryIngestor
    ingestor = SKYbraryIngestor()
    ingestor.run(max_articles=50)
"""

import json
import time
import hashlib
import requests
from datetime import date
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, IngestionError

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SKYBRARY_BASE_URL     = "https://skybrary.aero"
ACCIDENTS_LIST_URL    = f"{SKYBRARY_BASE_URL}/accidents-and-incidents"

# Target categories for AeroLex
TARGET_CATEGORIES = [
    "accidents-and-incidents",
]

# ── Data directory ────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path("data/raw/skybrary")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
HASH_FILE    = RAW_DATA_DIR / "article_hashes.json"
ARTICLE_INDEX = RAW_DATA_DIR / "article_index.json"


class SKYbraryIngestor:
    """
    Scrapes SKYbrary aviation safety articles using BeautifulSoup.

    Two-phase approach:
    Phase A — Scrape article listing → collect URLs
    Phase B — Scrape each article → extract text + metadata
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AeroLex-RAG/1.0 (Aviation Safety Research)",
            "Accept":     "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9"
        })
        self.hashes        = self._load_hashes()
        self.article_index = self._load_article_index()
        self.stats         = {
            "urls_found":    0,
            "scraped":       0,
            "skipped":       0,
            "failed":        0
        }
        logger.info("SKYbraryIngestor initialized")

    # ── Hash & Index Management ───────────────────────────────────────────────

    def _load_hashes(self) -> dict:
        if HASH_FILE.exists():
            with open(HASH_FILE) as f:
                return json.load(f)
        return {}

    def _save_hashes(self) -> None:
        with open(HASH_FILE, "w") as f:
            json.dump(self.hashes, f, indent=2)

    def _load_article_index(self) -> dict:
        if ARTICLE_INDEX.exists():
            with open(ARTICLE_INDEX) as f:
                return json.load(f)
        return {}

    def _save_article_index(self) -> None:
        with open(ARTICLE_INDEX, "w") as f:
            json.dump(self.article_index, f, indent=2)

    def _compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # ── Article URL Collection ────────────────────────────────────────────────

    def _get_article_urls(self, max_pages: int = 3) -> list[str]:
        """
        Collect article URLs from SKYbrary listing pages.

        SKYbrary uses pagination — each page has ~20 articles.
        URL pattern: /accidents-and-incidents?page=1

        Args:
            max_pages: Number of listing pages to scrape

        Returns:
            list: Article URLs
        """
        urls = []

        for page_num in range(0, max_pages):
            try:
                # Page 0 = first page (no ?page param needed)
                if page_num == 0:
                    url = ACCIDENTS_LIST_URL
                else:
                    url = f"{ACCIDENTS_LIST_URL}?page={page_num}"

                logger.info(f"Fetching article list page {page_num}: {url}")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")

                # Find all article links
                # SKYbrary article links pattern: /accidents-and-incidents/{slug}
                page_urls = []
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    if (
                        "/accidents-and-incidents/" in href
                        and href != "/accidents-and-incidents/"
                        and "?" not in href
                        and "#" not in href
                    ):
                        # Build full URL
                        if href.startswith("http"):
                            full_url = href
                        else:
                            full_url = SKYBRARY_BASE_URL + href

                        if full_url not in urls:
                            page_urls.append(full_url)

                urls.extend(page_urls)
                logger.info(
                    f"Page {page_num}: Found {len(page_urls)} article URLs | "
                    f"Total so far: {len(urls)}"
                )

                # Check if there's a next page
                next_link = soup.find("a", rel="next") or soup.find("a", string="Next")
                if not next_link and page_num > 0:
                    logger.info("No more pages found")
                    break

                time.sleep(1)  # Rate limiting

            except Exception as e:
                logger.warning(f"Failed to fetch listing page {page_num}: {e}")
                continue

        # Deduplicate
        urls = list(dict.fromkeys(urls))
        logger.info(f"Total unique article URLs collected: {len(urls)}")
        return urls

    # ── Article Scraping ──────────────────────────────────────────────────────

    def _scrape_article(self, url: str) -> Optional[dict]:
        """
        Scrape a single SKYbrary article.

        Extracts:
        - Title
        - Full article text
        - Categories/tags
        - Related articles
        - Summary/synopsis if available

        Args:
            url: Full article URL

        Returns:
            dict: Article record or None if failed
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # ── Extract title ─────────────────────────────────────────────
            title = ""
            title_tag = (
                soup.find("h1") or
                soup.find("title")
            )
            if title_tag:
                title = title_tag.get_text(strip=True)
                # Clean up " | SKYbrary" suffix
                title = title.replace(" | SKYbrary", "").strip()

            # ── Extract main article text ─────────────────────────────────
            # SKYbrary article content is in main content area
            text = ""
            content_area = (
                soup.find("article") or
                soup.find("div", class_="article-content") or
                soup.find("div", class_="node__content") or
                soup.find("main") or
                soup.find("div", role="main")
            )

            if content_area:
                # Remove navigation, ads, footer elements
                for unwanted in content_area.find_all(
                    ["nav", "footer", "script", "style", "aside"]
                ):
                    unwanted.decompose()

                text = content_area.get_text(separator=" ", strip=True)
            else:
                # Fallback — get body text
                body = soup.find("body")
                if body:
                    for unwanted in body.find_all(
                        ["nav", "footer", "script", "style", "header"]
                    ):
                        unwanted.decompose()
                    text = body.get_text(separator=" ", strip=True)

            # Clean up text
            import re
            text = re.sub(r'\s+', ' ', text).strip()

            # Skip if too short — likely a redirect or error page
            if len(text) < 100:
                logger.debug(f"Article too short, skipping: {url}")
                return None

            # ── Extract slug from URL ─────────────────────────────────────
            slug = url.rstrip("/").split("/")[-1]

            # ── Extract categories from URL path ──────────────────────────
            url_parts = url.replace(SKYBRARY_BASE_URL, "").strip("/").split("/")
            category  = url_parts[0] if url_parts else "general"

            # ── Extract meta description as summary ───────────────────────
            summary = ""
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc:
                summary = meta_desc.get("content", "").strip()

            record = {
                "article_id":     f"skybrary_{slug}",
                "title":          title,
                "url":            url,
                "slug":           slug,
                "category":       category,
                "summary":        summary,
                "text":           text[:50000],  # Cap at 50K chars
                "text_length":    len(text),
                "ingestion_date": date.today().isoformat(),
                "source":         "skybrary",
                "doc_type":       "safety_article"
            }

            return record

        except Exception as e:
            logger.warning(f"Failed to scrape article {url}: {e}")
            return None

    # ── Main Ingestion ────────────────────────────────────────────────────────

    def run(
        self,
        max_articles: int = 50,
        max_pages: int = 3
    ) -> dict:
        """
        Run full SKYbrary ingestion.

        Args:
            max_articles: Max articles to scrape (cost/time control)
            max_pages: Max listing pages to check for URLs

        Returns:
            dict: Ingestion stats
        """
        logger.info(
            f"Starting SKYbrary ingestion | "
            f"Max articles: {max_articles} | "
            f"Max pages: {max_pages}"
        )

        # Phase A: Collect article URLs
        article_urls = self._get_article_urls(max_pages=max_pages)
        self.stats["urls_found"] = len(article_urls)

        # Limit to max_articles
        article_urls = article_urls[:max_articles]
        logger.info(f"Processing {len(article_urls)} articles")

        # Phase B: Scrape each article
        for i, url in enumerate(article_urls):
            try:
                slug       = url.rstrip("/").split("/")[-1]
                article_id = f"skybrary_{slug}"

                # Quick hash check using URL as proxy
                # Full hash done after scraping
                if article_id in self.article_index:
                    # Already indexed — check if content changed
                    # For SKYbrary, articles rarely change
                    # Skip if already in index for efficiency
                    self.stats["skipped"] += 1
                    logger.debug(f"Already indexed: {slug}")
                    continue

                logger.info(f"Scraping [{i+1}/{len(article_urls)}]: {slug}")
                record = self._scrape_article(url)

                if record is None:
                    self.stats["failed"] += 1
                    continue

                # Hash check on content
                new_hash = self._compute_hash(record["text"][:1000])
                if self.hashes.get(article_id) == new_hash:
                    self.stats["skipped"] += 1
                    continue

                # Save
                self.hashes[article_id]        = new_hash
                self.article_index[article_id] = record
                self.stats["scraped"] += 1

                logger.info(
                    f"Article scraped: {record['title'][:50]} | "
                    f"Text: {record['text_length']} chars"
                )

                # Save every 10 articles
                if self.stats["scraped"] % 10 == 0:
                    self._save_hashes()
                    self._save_article_index()

                time.sleep(1)  # Polite rate limiting

            except Exception as e:
                handle_exception(e, context=f"SKYbraryIngestor.run() {url}")
                self.stats["failed"] += 1
                continue

        # Final save
        self._save_hashes()
        self._save_article_index()

        logger.info(
            f"SKYbrary ingestion complete | "
            f"URLs found: {self.stats['urls_found']} | "
            f"Scraped: {self.stats['scraped']} | "
            f"Skipped: {self.stats['skipped']} | "
            f"Failed: {self.stats['failed']}"
        )
        return self.stats


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Testing SKYbrary Ingestor ---\n")
    print("⚠️  Scraping 10 articles from skybrary.aero\n")

    ingestor = SKYbraryIngestor()
    stats    = ingestor.run(max_articles=10, max_pages=1)

    print(f"\n📊 Ingestion Stats:")
    print(f"   URLs found  : {stats['urls_found']}")
    print(f"   Scraped     : {stats['scraped']}")
    print(f"   Skipped     : {stats['skipped']}")
    print(f"   Failed      : {stats['failed']}")

    print(f"\n📋 Sample Articles:")
    for article_id, article in list(ingestor.article_index.items())[:3]:
        print(f"   [{article['category']}] {article['title'][:60]}")
        print(f"   URL: {article['url']}")
        print(f"   Text: {article['text_length']} chars")
        print(f"   Preview: {article['text'][:150]}...")
        print()

    print(f"Total articles indexed: {len(ingestor.article_index)}")
    print(f"\n✅ SKYbrary Ingestor working!")