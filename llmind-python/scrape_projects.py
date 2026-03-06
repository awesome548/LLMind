#!/usr/bin/env python3
"""
Scrape Media Architecture Biennale project pages.

Flow:
1) Fetch listing page (default: https://awards.mediaarchitecture.org/mab/projects/)
2) Extract project numbers from each `div.mab-card > a[href="/mab/project/{id}"]`
3) Visit up to --limit project pages and extract:
   - Name (from .titlepro excluding <small>)
   - Description (joined .col-sm-6 texts with junk filtered)
4) Write JSON

Usage:
  python scrape_projects.py scrape --limit 20
  python scrape_projects.py scrape --listing https://awards.mediaarchitecture.org/mab/projects/ --limit 5 -o out.json
"""

from __future__ import annotations

import json
import re
import time
from typing import List, Optional, Tuple
from urllib.parse import urlparse, urljoin

from tenacity import Retrying, stop_after_attempt, wait_exponential, retry_if_exception_type
import typer

import requests
from bs4 import BeautifulSoup

from config import settings
from utils.models import ProjectRecord
from pydantic import ValidationError

# -----------------------
# Configuration
# -----------------------

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MABProjectScraper/1.0; +https://example.org/)"
}
TIMEOUT = 30

SAVE_HTML_SNAPSHOT = False  # set True to include raw main HTML in JSON


# -----------------------
# HTTP helpers
# -----------------------

def fetch_html(url: str, session: requests.Session, retryer: Retrying) -> str:
    """Fetch URL with retries and return HTML, raising on non-200 responses."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported scheme in URL: {url}")
    if "awards.mediaarchitecture.org" not in parsed.netloc:
        raise ValueError(f"Unexpected domain for URL: {url}")

    def _do_request() -> str:
        resp = session.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.text

    # Use tenacity for retries on network-related errors and 5xx
    for attempt in retryer:
        with attempt:
            try:
                return _do_request()
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                # Retry only on 5xx; re-raise others
                if status and 500 <= status < 600:
                    typer.secho(f"HTTP {status} on {url}; retrying...", fg=typer.colors.YELLOW)
                    raise
                typer.secho(f"HTTP {status or 'error'} on {url}; not retrying.", fg=typer.colors.RED)
                raise
            except (requests.ConnectionError, requests.Timeout) as e:
                typer.secho(f"Network error on {url}: {e}; retrying...", fg=typer.colors.YELLOW)
                raise

    return "Success"

def build_retryer(max_attempts: int, initial_backoff: float, max_backoff: float) -> Retrying:
    return Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=initial_backoff, max=max_backoff),
        retry=retry_if_exception_type((requests.HTTPError, requests.ConnectionError, requests.Timeout)),
        reraise=True,
    )


# -----------------------
# Parsing helpers
# -----------------------

def extract_main_container(soup: BeautifulSoup):
    """Try to find a main content container to snapshot if requested."""
    for sel in ("main", "div.container", "div.container-fluid", "body"):
        node = soup.select_one(sel)
        if node:
            return node
    return soup  # fallback


def parse_listing_for_project_urls(listing_html: str, limit: Optional[int] = None) -> List[str]:
    """
    From the listing page, find each project card:
      <div class="mab-card ..."><a href="/mab/project/{id}">...</a></div>
    Return absolute URLs (BASE + href). Preserve on-page order. De-duplicate.
    If limit is 0, scrape all projects (no limit).
    """
    soup = BeautifulSoup(listing_html, "html.parser")
    project_urls: List[str] = []
    seen = set()

    # Regex for /mab/project/{number}
    href_re = re.compile(r"^/mab/project/(\d+)/?$")

    for card in soup.select("div.mab-card"):
        a = card.find("a", href=True)
        if not a:
            continue
        href = a["href"]
        m = href_re.match(href)
        if not m:
            continue
        abs_url = urljoin(settings.base_url, href)
        if abs_url in seen:
            continue
        seen.add(abs_url)
        project_urls.append(abs_url)
        # If limit is 0, scrape all projects (no limit)
        if limit is not None and limit > 0 and len(project_urls) >= limit:
            break

    return project_urls


def parse_project_page(url: str, html: str) -> ProjectRecord:
    soup = BeautifulSoup(html, "html.parser")

    # 1) Collect all .titlepro texts (exclude <small>)
    titles = []
    for t in soup.select(".titlepro"):
        for small in t.find_all("small"):
            small.extract()
        txt = t.get_text(" ", strip=True)
        if txt:
            titles.append(txt)
    titlepro = " | ".join(titles) if titles else ""

    # 2) Collect non-empty .col-sm-6 texts (skip empty / "None" / <a href="None">None</a>)
    cols_sm_6 = []
    cols_sm_4 = []
    for c in soup.select(".col-sm-6"):
        parts = []
        for child in c.find_all(["p", "h5"], recursive=True):
            text = child.get_text(" ", strip=True)
            if not text or text.lower() == "none":
                continue
            if text.lower() == "none":
                continue
            
                        # Remove URLs from the extracted text
            text_no_urls = re.sub(r"http\S+|www\.\S+", "", text).strip()
            if text_no_urls:
                parts.append(text_no_urls)

        if parts:
            cols_sm_6.append(" ".join(parts))
    col_sm_6 = " || ".join(cols_sm_6)

    for c in soup.select(".col-sm-4"):
        parts = []
        start = c.find("h5", class_="mediumkur", string=lambda s: s and s.strip() == "Descriptions")
        if not start:
            continue

        for sib in start.find_next_siblings():
            if sib.name == "h5":
                break  # stop at the next header
            if sib.name != "p":
                continue  # only collect p tags

            text = sib.get_text(" ", strip=True)
            if not text or text.lower() == "none":
                continue

            # Remove URLs
            text_no_urls = re.sub(r"http\S+|www\.\S+", "", text).strip()
            if text_no_urls:
                parts.append(text_no_urls)

        if parts:
            cols_sm_4.append("\n".join(parts))
    col_sm_4 = " || ".join(cols_sm_4)

    html_main = None
    if SAVE_HTML_SNAPSHOT:
        html_main = str(extract_main_container(soup))

    image_href: Optional[str] = None
    gallery_img = soup.select_one("img.gallery.img-fluid.img-responsive")
    if gallery_img:
        parent_link = gallery_img.find_parent("a", href=True)
        if parent_link and parent_link["href"]:
            image_href = urljoin(url, parent_link["href"])
        else:
            src = gallery_img.get("src")
            if src:
                image_href = urljoin(url, src)

    return ProjectRecord(
        url=url,
        Name=titlepro,
        Descriptions=col_sm_6,
        Details=col_sm_4,
        image_href=image_href,
        html_main=html_main,
    )


# -----------------------
# I/O helpers
# -----------------------

def write_json(path: str, rows: List[ProjectRecord]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([
            r.model_dump() if hasattr(r, "model_dump") else r.dict()
            for r in rows
        ], f, ensure_ascii=False, indent=2)


# -----------------------
# CLI
# -----------------------

app = typer.Typer(add_completion=False)

def run_scrape(
    limit: Optional[int] = None,
    delay: float = 0.8,
    max_attempts: int = 4,
    initial_backoff: float = 1.0,
    max_backoff: float = 10.0,
) -> list[ProjectRecord]:
    """Scrape project pages and return records. No Typer dependency."""
    session = requests.Session()
    retryer = build_retryer(max_attempts=max_attempts, initial_backoff=initial_backoff, max_backoff=max_backoff)

    listing_url = settings.default_listing_url
    typer.echo(f"Fetching listing: {listing_url}")
    listing_html = fetch_html(listing_url, session=session, retryer=retryer)
    project_urls = parse_listing_for_project_urls(listing_html, limit=limit)
    if not project_urls:
        typer.secho("No project URLs found on listing page.", fg=typer.colors.RED)
        return []

    typer.secho(f"Discovered {len(project_urls)} project URL(s).", fg=typer.colors.GREEN)

    results: list[ProjectRecord] = []
    for i, url in enumerate(project_urls, 1):
        try:
            html = fetch_html(url, session=session, retryer=retryer)
            rec = parse_project_page(url, html)
            results.append(rec)
            typer.secho(f"[{i}/{len(project_urls)}] OK {url} -> Name='{rec.Name[:80]}'", fg=typer.colors.GREEN)
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", "HTTPError")
            typer.secho(f"[{i}/{len(project_urls)}] HTTP error {code} on {url}", fg=typer.colors.RED)
        except ValidationError as e:
            typer.secho(f"[{i}/{len(project_urls)}] Data validation error on {url}: {e}", fg=typer.colors.RED)
        except Exception as e:
            typer.secho(f"[{i}/{len(project_urls)}] Unhandled error on {url}: {e}", fg=typer.colors.RED)
        finally:
            time.sleep(delay)

    return results


@app.command("scrape")
def scrape(
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-n",
        min=0,
        help="Maximum number of project pages to scrape (in listing order). Use 0 to scrape all projects."
    ),
    json_path: Optional[str] = typer.Option(
        "media_architecture_projects.json",
        "--out",
        "-o",
        help="JSON file name (not directory; will be placed under DATA_DIR)."
    ),
    delay: float = typer.Option(
        0.8,
        "--delay",
        "-d",
        help="Polite delay (seconds) between requests)."
    ),
    max_attempts: int = typer.Option(
        4,
        "--retries",
        "-r",
        min=1,
        help="Maximum retry attempts per request."
    ),
    initial_backoff: float = typer.Option(
        1.0,
        "--backoff",
        help="Initial backoff (seconds) for exponential retry."
    ),
    max_backoff: float = typer.Option(
        10.0,
        "--max-backoff",
        help="Maximum backoff (seconds) for exponential retry."
    ),
) -> list[ProjectRecord]:
    """Discover project URLs from the listing page, then scrape each project."""
    results = run_scrape(
        limit=limit,
        delay=delay,
        max_attempts=max_attempts,
        initial_backoff=initial_backoff,
        max_backoff=max_backoff,
    )

    if not results:
        raise typer.Exit(code=2)

    if json_path:
        write_json(str(settings.data_dir / json_path), results)
        typer.secho(f"Saved JSON -> {json_path}", fg=typer.colors.GREEN)

    return results

if __name__ == "__main__":
    app()
