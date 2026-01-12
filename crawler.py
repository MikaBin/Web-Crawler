"""Simple web crawler for extracting guide content from a URL."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen


@dataclass
class PageData:
    url: str
    title: str
    text: str
    links: list[str] = field(default_factory=list)


class HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._texts: list[str] = []
        self._links: list[str] = []
        self._in_title = False
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value:
                    self._links.append(value)
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
        self._texts.append(data)

    def get_text(self) -> str:
        return " ".join(text.strip() for text in self._texts if text.strip())

    def get_title(self) -> str:
        return " ".join(part.strip() for part in self._title_parts if part.strip())

    def get_links(self) -> list[str]:
        return self._links


class SimpleCrawler:
    def __init__(
        self,
        start_urls: Iterable[str],
        max_pages: int = 25,
        max_depth: int = 2,
        same_domain_only: bool = True,
        delay_seconds: float = 0.5,
        user_agent: str = "WebCrawlerBot/1.0",
    ) -> None:
        self.start_urls = list(start_urls)
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.same_domain_only = same_domain_only
        self.delay_seconds = delay_seconds
        self.user_agent = user_agent
        self._visited: set[str] = set()

    def crawl(self) -> list[PageData]:
        results: list[PageData] = []
        queue: deque[tuple[str, int]] = deque(
            (normalize_url(url), 0) for url in self.start_urls
        )
        while queue and len(results) < self.max_pages:
            url, depth = queue.popleft()
            if url in self._visited or depth > self.max_depth:
                continue
            self._visited.add(url)
            try:
                page_data = self._fetch_page(url)
            except Exception:
                continue
            results.append(page_data)
            if depth < self.max_depth:
                for link in page_data.links:
                    normalized = normalize_url(urljoin(url, link))
                    if not normalized or normalized in self._visited:
                        continue
                    if self.same_domain_only and not is_same_domain(url, normalized):
                        continue
                    queue.append((normalized, depth + 1))
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
        return results

    def _fetch_page(self, url: str) -> PageData:
        request = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=15) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                raise ValueError("Non-HTML content")
            body = response.read().decode("utf-8", errors="ignore")
        parser = HTMLTextExtractor()
        parser.feed(body)
        return PageData(
            url=url,
            title=parser.get_title(),
            text=parser.get_text(),
            links=parser.get_links(),
        )


def normalize_url(url: str) -> str:
    if not url:
        return ""
    split = urlsplit(url)
    if split.scheme not in {"http", "https"}:
        return ""
    normalized = urlunsplit(
        (split.scheme, split.netloc.lower(), split.path or "/", split.query, "")
    )
    return normalized


def is_same_domain(base_url: str, target_url: str) -> bool:
    return urlsplit(base_url).netloc.lower() == urlsplit(target_url).netloc.lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl guide pages from a URL")
    parser.add_argument("start_url", help="Starting URL to crawl")
    parser.add_argument("--max-pages", type=int, default=25)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--allow-external", action="store_true")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--output", default="crawl_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crawler = SimpleCrawler(
        [args.start_url],
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        same_domain_only=not args.allow_external,
        delay_seconds=args.delay,
    )
    results = crawler.crawl()
    payload = [page.__dict__ for page in results]
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Saved {len(payload)} pages to {args.output}")


if __name__ == "__main__":
    main()
