"""
Parser for Help Center articles from CSV.
"""

import re
from dataclasses import dataclass, field

import pandas as pd
from bs4 import BeautifulSoup

from ..config import HELP_CSV
from .features import ContentFeatureDetector, ContentFeatures


@dataclass
class Article:
    """Represents a Help Center article."""

    title: str
    md_content: str
    raw_html: str
    features: ContentFeatures = field(default_factory=ContentFeatures)


class HelpCenterParser:
    """Parse Help Center articles from CSV and extract clean text in md format."""

    def __init__(self, csv_path: str | None = None):
        self.csv_path = csv_path or str(HELP_CSV)

    @staticmethod
    def _remove_noise_tags(
        soup: BeautifulSoup, tags: list[str] = ["style", "img", "script", "iframe"]
    ):
        """Removes noisy tags like style, img, script, iframe."""
        for tag in soup(tags):
            tag.decompose()

    @staticmethod
    def _process_headers(soup: BeautifulSoup):
        """Converts header tags (h1-h6) into structured text with markers."""
        for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            if not header.parent:
                continue
            level = int(header.name[1])
            prefix = "#" * level
            header.insert_before(f"\n{prefix} {header.get_text(strip=True)}\n")
            header.decompose()

    @staticmethod
    def _process_lists(soup: BeautifulSoup):
        """Converts ordered and unordered lists into structured text."""
        for ul in soup.find_all("ul"):
            if not ul.parent:
                continue
            items = [f"- {li.get_text(strip=True)}" for li in ul.find_all("li")]
            ul.insert_before("\n" + "\n".join(items) + "\n")
            ul.decompose()

        for ol in soup.find_all("ol"):
            if not ol.parent:
                continue
            items = [
                f"{i + 1}. {li.get_text(strip=True)}"
                for i, li in enumerate(ol.find_all("li"))
            ]
            ol.insert_before("\n" + "\n".join(items) + "\n")
            ol.decompose()

    @staticmethod
    def _clean_text_formatting(text: str) -> str:
        """Cleans multiple spaces and empty lines in the text."""
        # 1. Collapse multiple spaces and tabs on each line
        lines = []
        for line in text.splitlines():
            cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
            lines.append(cleaned_line)
        text = "\n".join(lines)

        # 2. Collapse 3 or more consecutive newlines to exactly 2 newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def parse_html_to_md(raw_html: str) -> str:
        """Cleans HTML and converts it to structured text in md format"""
        soup = BeautifulSoup(raw_html, "html.parser")

        HelpCenterParser._remove_noise_tags(soup)
        HelpCenterParser._process_headers(soup)
        HelpCenterParser._process_lists(soup)

        text = soup.get_text(separator="\n", strip=False)

        return HelpCenterParser._clean_text_formatting(text)

    def load_articles(self) -> list[Article]:
        """Load and parse all articles from CSV."""
        df = pd.read_csv(self.csv_path)

        articles = []
        for _, row in df.iterrows():
            title = row["title"]
            raw_html = row["article_content"]
            md_content = HelpCenterParser.parse_html_to_md(raw_html)

            # Detect content features for metadata enrichment
            features = ContentFeatureDetector.detect(raw_html, md_content)

            articles.append(
                Article(
                    title=title,
                    md_content=md_content,
                    raw_html=raw_html,
                    features=features,
                )
            )

        return articles

    def __repr__(self) -> str:
        return f"HelpCenterParser(csv_path='{self.csv_path}')"
