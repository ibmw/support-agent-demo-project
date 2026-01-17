"""
Content feature detection for Help Center articles.

Detects characteristics of content to enrich metadata for better retrieval.
"""

import re
from dataclasses import dataclass


@dataclass
class ContentFeatures:
    """Features detected from article content."""

    has_steps: bool = False  # Step-by-step guide
    is_faq: bool = False  # FAQ content
    has_code: bool = False  # Contains code snippets
    is_integration: bool = False  # Integration/setup guide
    is_troubleshooting: bool = False  # Troubleshooting content
    has_pricing: bool = False  # Pricing/billing related
    content_type: str = "general"  # Inferred content type

    def to_dict(self) -> dict:
        """Convert to dictionary for metadata storage."""
        return {
            "has_steps": self.has_steps,
            "is_faq": self.is_faq,
            "has_code": self.has_code,
            "is_integration": self.is_integration,
            "is_troubleshooting": self.is_troubleshooting,
            "has_pricing": self.has_pricing,
            "content_type": self.content_type,
        }


class ContentFeatureDetector:
    """Detects content features from HTML and cleaned text."""

    # Step-by-step patterns
    STEP_PATTERNS = [
        r"\b(?:step|étape)s?\s*\d+",  # "Step 1", "Étape 2"
        r"\d+\.\s+(?:go to|click|enter|navigate|select|open)",  # Numbered action lists
        r"(?:how to|comment)\s+",  # "How to" guides
        r"(?:first|then|next|finally),?\s+",  # Sequential instructions
    ]

    # FAQ patterns
    FAQ_PATTERNS = [
        r"\b(?:faq|frequently asked)\b",
        r"\?[\s\n]+(?:answer|response|solution):",
        r"^q:\s+|^a:\s+",  # Q&A format
    ]

    # Code patterns (check in raw HTML)
    CODE_PATTERNS = [
        r"<code[^>]*>",
        r"<pre[^>]*>",
        r"```",
        r"<script[^>]*>",
    ]

    # Integration patterns
    INTEGRATION_PATTERNS = [
        r"\b(?:integration|integrate|connect|authorize|oauth)\b",
        r"\b(?:install|setup|configure|enable)\b.*\b(?:app|plugin|extension)\b",
        r"\bapi\s*key\b",
        r"\b(?:webhook|endpoint)\b",
    ]

    # Troubleshooting patterns
    TROUBLESHOOTING_PATTERNS = [
        r"\b(?:troubleshoot|fix|solve|resolve|issue|problem|error)\b",
        r"\b(?:not working|doesn't work|won't|can't|unable)\b",
        r"\bif you (?:see|get|encounter|experience)\b",
    ]

    # Pricing patterns
    PRICING_PATTERNS = [
        r"\b(?:pricing|price|cost|plan|subscription|billing)\b",
        r"\$\d+|\d+\s*(?:usd|eur|gbp)",
        r"\b(?:free|premium|enterprise|starter)\s*(?:plan|tier)?\b",
    ]

    @classmethod
    def detect(cls, raw_html: str, cleaned_text: str) -> ContentFeatures:
        """
        Detect content features from HTML and cleaned text.

        Args:
            raw_html: Original HTML content
            cleaned_text: Cleaned markdown/text content

        Returns:
            ContentFeatures with detected characteristics
        """
        text_lower = cleaned_text.lower()
        html_lower = raw_html.lower()

        features = ContentFeatures(
            has_steps=cls._match_any(text_lower, cls.STEP_PATTERNS),
            is_faq=cls._match_any(text_lower, cls.FAQ_PATTERNS),
            has_code=cls._match_any(html_lower, cls.CODE_PATTERNS),
            is_integration=cls._match_any(text_lower, cls.INTEGRATION_PATTERNS),
            is_troubleshooting=cls._match_any(text_lower, cls.TROUBLESHOOTING_PATTERNS),
            has_pricing=cls._match_any(text_lower, cls.PRICING_PATTERNS),
        )

        # Infer content type based on features
        features.content_type = cls._infer_content_type(features)

        return features

    @classmethod
    def _match_any(cls, text: str, patterns: list[str]) -> bool:
        """Check if any pattern matches the text."""
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    @classmethod
    def _infer_content_type(cls, features: ContentFeatures) -> str:
        """Infer the primary content type from features."""
        # Priority-based inference
        if features.is_faq:
            return "faq"
        if features.is_troubleshooting:
            return "troubleshooting"
        if features.is_integration:
            return "integration"
        if features.has_steps:
            return "how-to"
        if features.has_code:
            return "technical"
        if features.has_pricing:
            return "pricing"
        return "general"
