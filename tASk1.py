"""
=============================================================
  WEB SCRAPING TOOLKIT
  Techniques: BeautifulSoup, Scrapy, Requests, CSV/JSON Export
=============================================================
"""

# ─────────────────────────────────────────────
# SECTION 1: DEPENDENCIES
# ─────────────────────────────────────────────
# pip install requests beautifulsoup4 lxml scrapy pandas

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import csv
import time
import logging
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, asdict
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SECTION 2: BASE HTTP SESSION (with headers & retries)
# ─────────────────────────────────────────────

def make_session(retries: int = 3) -> requests.Session:
    """Create a reusable HTTP session with a browser-like User-Agent."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    })
    # Mount a retry adapter
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    retry = Retry(total=retries, backoff_factor=1,
                  status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://",  HTTPAdapter(max_retries=retry))
    return session


def fetch_page(url: str, session: requests.Session, delay: float = 1.0) -> Optional[BeautifulSoup]:
    """Fetch a URL and return a BeautifulSoup object, or None on failure."""
    try:
        time.sleep(delay)                         # polite crawl delay
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except requests.RequestException as e:
        log.error(f"Failed to fetch {url}: {e}")
        return None


# ─────────────────────────────────────────────
# SECTION 3: DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class Article:
    title:   str
    url:     str
    summary: str
    date:    str

@dataclass
class Product:
    name:     str
    price:    str
    rating:   str
    url:      str
    image:    str

@dataclass
class JobListing:
    title:    str
    company:  str
    location: str
    url:      str


# ─────────────────────────────────────────────
# SECTION 4: BEAUTIFULSOUP SCRAPERS
# ─────────────────────────────────────────────

class NewsScraper:
    """
    Scrape article listings from a news-style page.
    Targets: <article> tags → title (h2/h3), summary (p), date (time), link (a).
    """

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session  = make_session()

    def parse_article(self, tag) -> Optional[Article]:
        title_tag = tag.find(["h2", "h3"])
        link_tag  = tag.find("a", href=True)
        para_tag  = tag.find("p")
        time_tag  = tag.find("time")

        if not (title_tag and link_tag):
            return None

        return Article(
            title   = title_tag.get_text(strip=True),
            url     = urljoin(self.base_url, link_tag["href"]),
            summary = para_tag.get_text(strip=True) if para_tag else "",
            date    = (time_tag.get("datetime") or time_tag.get_text(strip=True))
                      if time_tag else "",
        )

    def scrape(self, path: str = "/") -> List[Article]:
        url  = urljoin(self.base_url, path)
        soup = fetch_page(url, self.session)
        if not soup:
            return []

        articles = []
        for tag in soup.find_all("article"):
            item = self.parse_article(tag)
            if item:
                articles.append(item)

        log.info(f"Scraped {len(articles)} articles from {url}")
        return articles


class ProductScraper:
    """
    Scrape product cards from an e-commerce listing page.
    Adapts selectors via a config dict so you can reuse across sites.
    """

    DEFAULT_SELECTORS = {
        "card":   ".product-card",
        "name":   ".product-title",
        "price":  ".price",
        "rating": ".rating",
        "link":   "a",
        "image":  "img",
    }

    def __init__(self, base_url: str, selectors: dict = None):
        self.base_url  = base_url
        self.sel       = {**self.DEFAULT_SELECTORS, **(selectors or {})}
        self.session   = make_session()

    def parse_card(self, card) -> Optional[Product]:
        def text(selector):
            el = card.select_one(selector)
            return el.get_text(strip=True) if el else ""

        link = card.select_one(self.sel["link"])
        img  = card.select_one(self.sel["image"])

        return Product(
            name   = text(self.sel["name"]),
            price  = text(self.sel["price"]),
            rating = text(self.sel["rating"]),
            url    = urljoin(self.base_url, link["href"]) if link else "",
            image  = img.get("src", "") if img else "",
        )

    def scrape(self, path: str = "/", pages: int = 1) -> List[Product]:
        products = []
        for page in range(1, pages + 1):
            url  = f"{self.base_url}{path}?page={page}"
            soup = fetch_page(url, self.session)
            if not soup:
                break
            cards = soup.select(self.sel["card"])
            for card in cards:
                item = self.parse_card(card)
                if item and item.name:
                    products.append(item)
            log.info(f"Page {page}: {len(cards)} cards found")
        return products


class TableScraper:
    """
    Extract all HTML <table> elements from a page into Pandas DataFrames.
    Ideal for Wikipedia tables, government data portals, financial sites.
    """

    def __init__(self):
        self.session = make_session()

    def scrape(self, url: str) -> List[pd.DataFrame]:
        soup = fetch_page(url, self.session)
        if not soup:
            return []

        tables = soup.find_all("table")
        frames = []
        for i, tbl in enumerate(tables):
            rows, headers = [], []
            header_row = tbl.find("tr")
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]

            for tr in tbl.find_all("tr")[1:]:
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)

            if rows:
                df = pd.DataFrame(rows, columns=headers[:len(rows[0])] if headers else None)
                frames.append(df)
                log.info(f"Table {i+1}: {df.shape[0]} rows × {df.shape[1]} cols")

        return frames


# ─────────────────────────────────────────────
# SECTION 5: MULTI-PAGE LINK FOLLOWER
# ─────────────────────────────────────────────

class PaginatedScraper:
    """
    Follow 'Next' pagination links automatically and collect all items.
    Pass a parse_fn(soup) -> list callback for flexibility.
    """

    def __init__(self, base_url: str, next_selector: str = "a[rel='next']"):
        self.base_url      = base_url
        self.next_selector = next_selector
        self.session       = make_session()

    def scrape(self, start_path: str, parse_fn, max_pages: int = 10) -> list:
        url      = urljoin(self.base_url, start_path)
        all_data = []
        visited  = set()

        for _ in range(max_pages):
            if url in visited:
                break
            visited.add(url)

            soup = fetch_page(url, self.session)
            if not soup:
                break

            page_data = parse_fn(soup)
            all_data.extend(page_data)
            log.info(f"Collected {len(page_data)} items from {url}")

            next_link = soup.select_one(self.next_selector)
            if not next_link:
                break
            url = urljoin(self.base_url, next_link["href"])

        return all_data


# ─────────────────────────────────────────────
# SECTION 6: SCRAPY SPIDER (standalone file content)
# ─────────────────────────────────────────────

SCRAPY_SPIDER_CODE = '''
# Save as: quotes_spider.py
# Run with: scrapy runspider quotes_spider.py -o quotes.csv

import scrapy

class QuotesSpider(scrapy.Spider):
    name  = "quotes"
    start_urls = ["https://quotes.toscrape.com/"]

    custom_settings = {
        "DOWNLOAD_DELAY": 1,          # 1 second between requests
        "ROBOTSTXT_OBEY": True,       # respect robots.txt
        "FEEDS": {"quotes.csv": {"format": "csv", "overwrite": True}},
    }

    def parse(self, response):
        for quote in response.css("div.quote"):
            yield {
                "text":   quote.css("span.text::text").get(),
                "author": quote.css("small.author::text").get(),
                "tags":   ", ".join(quote.css("a.tag::text").getall()),
            }

        # Follow pagination
        next_page = response.css("li.next a::attr(href)").get()
        if next_page:
            yield response.follow(next_page, self.parse)
'''

SCRAPY_BOOKS_SPIDER = '''
# Save as: books_spider.py
# Run with: scrapy runspider books_spider.py -o books.json

import scrapy

class BooksSpider(scrapy.Spider):
    name = "books"
    start_urls = ["https://books.toscrape.com/catalogue/page-1.html"]

    custom_settings = {
        "DOWNLOAD_DELAY": 1,
        "ROBOTSTXT_OBEY": True,
    }

    def parse(self, response):
        for book in response.css("article.product_pod"):
            yield {
                "title":  book.css("h3 a::attr(title)").get(),
                "price":  book.css("p.price_color::text").get(),
                "rating": book.css("p.star-rating::attr(class)").get().split()[-1],
                "url":    response.urljoin(book.css("h3 a::attr(href)").get()),
            }

        next_page = response.css("li.next a::attr(href)").get()
        if next_page:
            yield response.follow(next_page, self.parse)
'''


# ─────────────────────────────────────────────
# SECTION 7: DATA EXPORT UTILITIES
# ─────────────────────────────────────────────

class DataExporter:
    """Export a list of dataclass instances to CSV, JSON, or Excel."""

    @staticmethod
    def to_csv(data: list, path: str):
        if not data:
            log.warning("No data to export.")
            return
        rows = [asdict(d) for d in data]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"CSV saved → {path}  ({len(rows)} rows)")

    @staticmethod
    def to_json(data: list, path: str):
        rows = [asdict(d) for d in data]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        log.info(f"JSON saved → {path}  ({len(rows)} records)")

    @staticmethod
    def to_excel(data: list, path: str):
        df = pd.DataFrame([asdict(d) for d in data])
        df.to_excel(path, index=False)
        log.info(f"Excel saved → {path}  ({len(df)} rows)")

    @staticmethod
    def tables_to_excel(frames: List[pd.DataFrame], path: str):
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for i, df in enumerate(frames):
                df.to_excel(writer, sheet_name=f"Table_{i+1}", index=False)
        log.info(f"Excel saved → {path}  ({len(frames)} sheets)")


# ─────────────────────────────────────────────
# SECTION 8: DEMO RUNNER (uses public test sites)
# ─────────────────────────────────────────────

def demo_quotes():
    """Scrape quotes.toscrape.com using table scraper approach."""
    log.info("=== DEMO: Quotes via TableScraper ===")
    session = make_session()
    soup = fetch_page("https://quotes.toscrape.com/", session)
    if not soup:
        return []

    quotes = []
    for div in soup.select("div.quote"):
        quotes.append({
            "text":   div.select_one("span.text").get_text(strip=True),
            "author": div.select_one("small.author").get_text(strip=True),
            "tags":   ", ".join(t.get_text(strip=True) for t in div.select("a.tag")),
        })
    log.info(f"Found {len(quotes)} quotes")
    return quotes


def demo_books():
    """Scrape book titles and prices from books.toscrape.com."""
    log.info("=== DEMO: Books Catalogue ===")
    session = make_session()
    soup = fetch_page("https://books.toscrape.com/", session)
    if not soup:
        return []

    STAR_MAP = {"One":1,"Two":2,"Three":3,"Four":4,"Five":5}
    books = []
    for article in soup.select("article.product_pod"):
        title  = article.select_one("h3 a")["title"]
        price  = article.select_one("p.price_color").get_text(strip=True)
        cls    = article.select_one("p.star-rating")["class"]
        rating = STAR_MAP.get(cls[-1], 0)
        books.append({"title": title, "price": price, "rating": rating})

    log.info(f"Found {len(books)} books")
    return books


def demo_wikipedia_table():
    """Extract a data table from a Wikipedia page."""
    log.info("=== DEMO: Wikipedia Table Extraction ===")
    ts  = TableScraper()
    url = "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"
    frames = ts.scrape(url)
    if frames:
        log.info(f"First table preview:\n{frames[0].head()}")
    return frames


# ─────────────────────────────────────────────
# SECTION 9: MAIN ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os
    OUT = "scraped_output"
    os.makedirs(OUT, exist_ok=True)

    exporter = DataExporter()

    # --- Demo 1: Quotes ---
    quotes = demo_quotes()
    if quotes:
        df_q = pd.DataFrame(quotes)
        df_q.to_csv(f"{OUT}/quotes.csv", index=False)
        df_q.to_json(f"{OUT}/quotes.json", orient="records", indent=2)
        print("\n📰 QUOTES SAMPLE:")
        print(df_q.head(3).to_string(index=False))

    # --- Demo 2: Books ---
    books = demo_books()
    if books:
        df_b = pd.DataFrame(books)
        df_b.to_csv(f"{OUT}/books.csv", index=False)
        print("\n📚 BOOKS SAMPLE:")
        print(df_b.head(5).to_string(index=False))

    # --- Save Scrapy spiders for reference ---
    with open(f"{OUT}/quotes_spider.py", "w") as f:
        f.write(SCRAPY_SPIDER_CODE.strip())
    with open(f"{OUT}/books_spider.py", "w") as f:
        f.write(SCRAPY_BOOKS_SPIDER.strip())

    print(f"\n✅ All outputs saved to ./{OUT}/")
    print("   quotes.csv | quotes.json | books.csv")
    print("   quotes_spider.py | books_spider.py  (run with Scrapy)")