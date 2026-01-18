"""Unit tests for the Help Center HTML parser."""

from support_agent.indexing.parser import Article, HelpCenterParser


class TestParseHtmlToMd:
    """Tests for HelpCenterParser.parse_html_to_md()"""

    def test_removes_style_tags(self):
        """Style tags should be completely removed."""
        html = "<style>body { color: red; }</style><p>Hello</p>"
        result = HelpCenterParser.parse_html_to_md(html)
        assert "color" not in result
        assert "style" not in result.lower()
        assert "Hello" in result

    def test_removes_img_tags(self):
        """Image tags should be stripped."""
        html = '<p>Before</p><img src="test.png" alt="test"/><p>After</p>'
        result = HelpCenterParser.parse_html_to_md(html)
        assert "img" not in result.lower()
        assert "src" not in result
        assert "Before" in result
        assert "After" in result

    def test_removes_script_tags(self):
        """Script tags and content should be removed."""
        html = "<script>alert('xss')</script><p>Content</p>"
        result = HelpCenterParser.parse_html_to_md(html)
        assert "script" not in result.lower()
        assert "alert" not in result
        assert "Content" in result

    def test_removes_iframe_tags(self):
        """Iframe tags should be removed."""
        html = '<iframe src="https://example.com"></iframe><p>Text</p>'
        result = HelpCenterParser.parse_html_to_md(html)
        assert "iframe" not in result.lower()
        assert "Text" in result

    def test_converts_h1_to_markdown(self):
        """H1 tags should become # headings."""
        html = "<h1>Main Title</h1><p>Content here.</p>"
        result = HelpCenterParser.parse_html_to_md(html)
        assert "# Main Title" in result

    def test_converts_h2_to_markdown(self):
        """H2 tags should become ## headings."""
        html = "<h2>Section</h2>"
        result = HelpCenterParser.parse_html_to_md(html)
        assert "## Section" in result

    def test_converts_all_header_levels(self):
        """All header levels h1-h6 should be converted."""
        html = """
        <h1>H1</h1>
        <h2>H2</h2>
        <h3>H3</h3>
        <h4>H4</h4>
        <h5>H5</h5>
        <h6>H6</h6>
        """
        result = HelpCenterParser.parse_html_to_md(html)
        assert "# H1" in result
        assert "## H2" in result
        assert "### H3" in result
        assert "#### H4" in result
        assert "##### H5" in result
        assert "###### H6" in result

    def test_converts_unordered_list(self):
        """UL/LI should become dash-prefixed items."""
        html = "<ul><li>First</li><li>Second</li></ul>"
        result = HelpCenterParser.parse_html_to_md(html)
        assert "- First" in result
        assert "- Second" in result

    def test_converts_ordered_list(self):
        """OL/LI should become numbered items."""
        html = "<ol><li>Step one</li><li>Step two</li><li>Step three</li></ol>"
        result = HelpCenterParser.parse_html_to_md(html)
        assert "1. Step one" in result
        assert "2. Step two" in result
        assert "3. Step three" in result

    def test_preserves_paragraph_text(self):
        """Paragraph content should be preserved."""
        html = "<p>This is important content.</p>"
        result = HelpCenterParser.parse_html_to_md(html)
        assert "This is important content." in result

    def test_cleans_multiple_spaces(self):
        """Multiple consecutive spaces should be collapsed."""
        html = "<p>Too    many     spaces</p>"
        result = HelpCenterParser.parse_html_to_md(html)
        # Should not have more than one space
        assert "    " not in result
        assert "Too many spaces" in result

    def test_cleans_multiple_newlines(self):
        """Multiple newlines should be collapsed to max two."""
        html = "<p>Line 1</p>\n\n\n\n<p>Line 2</p>"
        result = HelpCenterParser.parse_html_to_md(html)
        # Count max consecutive newlines
        assert "\n\n\n" not in result

    def test_strips_whitespace(self):
        """Result should be stripped of leading/trailing whitespace."""
        html = "   <p>Content</p>   "
        result = HelpCenterParser.parse_html_to_md(html)
        assert result == result.strip()

    def test_empty_html(self):
        """Empty HTML should return empty string."""
        result = HelpCenterParser.parse_html_to_md("")
        assert result == ""

    def test_html_with_only_noise_tags(self):
        """HTML with only removable tags should return empty."""
        html = "<style>css</style><script>js</script><img src='x'/>"
        result = HelpCenterParser.parse_html_to_md(html)
        assert result.strip() == ""

    def test_nested_elements(self):
        """Nested elements should be handled correctly."""
        html = "<div><p>Outer <strong>bold</strong> text</p></div>"
        result = HelpCenterParser.parse_html_to_md(html)
        assert "bold" in result
        assert "Outer" in result
        assert "text" in result

    def test_complex_html_structure(self, sample_html):
        """Full sample HTML should parse correctly."""
        result = HelpCenterParser.parse_html_to_md(sample_html)
        assert "# Getting Started" in result
        assert "Welcome to our Help Center." in result
        assert "## Steps" in result
        assert "1. Go to Settings" in result
        assert "2. Click on Integrations" in result
        assert "## FAQ" in result
        # img tag should be removed
        assert "example.png" not in result


class TestRemoveNoiseTags:
    """Tests for _remove_noise_tags static method."""

    def test_default_tags_removed(self):
        """Default noise tags (style, img, script, iframe) are removed."""
        from bs4 import BeautifulSoup

        html = """
        <div>
            <style>.class {}</style>
            <img src="x">
            <script>code</script>
            <iframe></iframe>
            <p>Keep this</p>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        HelpCenterParser._remove_noise_tags(soup)
        assert soup.find("style") is None
        assert soup.find("img") is None
        assert soup.find("script") is None
        assert soup.find("iframe") is None
        assert soup.find("p") is not None

    def test_custom_tags_removed(self):
        """Custom tag list should work."""
        from bs4 import BeautifulSoup

        html = "<div><nav>Nav</nav><footer>Footer</footer><p>Content</p></div>"
        soup = BeautifulSoup(html, "html.parser")
        HelpCenterParser._remove_noise_tags(soup, tags=["nav", "footer"])
        assert soup.find("nav") is None
        assert soup.find("footer") is None
        assert soup.find("p") is not None


class TestCleanTextFormatting:
    """Tests for _clean_text_formatting static method."""

    def test_collapses_spaces(self):
        """Multiple spaces become single space."""
        result = HelpCenterParser._clean_text_formatting("a    b")
        assert result == "a b"

    def test_collapses_newlines(self):
        """Multiple spaces/whitespace become single space (newlines become spaces)."""
        # Note: _clean_text_formatting collapses ALL whitespace to single space
        result = HelpCenterParser._clean_text_formatting("a\n\n\n\nb")
        assert result == "a b"

    def test_strips_text(self):
        """Text is stripped."""
        result = HelpCenterParser._clean_text_formatting("  text  ")
        assert result == "text"


class TestHelpCenterParserInit:
    """Tests for parser initialization."""

    def test_default_csv_path(self):
        """Parser uses default CSV path from config."""
        from support_agent.config import HELP_CSV

        parser = HelpCenterParser()
        assert parser.csv_path == str(HELP_CSV)

    def test_custom_csv_path(self):
        """Parser accepts custom CSV path."""
        parser = HelpCenterParser(csv_path="/custom/path.csv")
        assert parser.csv_path == "/custom/path.csv"

    def test_repr(self):
        """Parser has useful string representation."""
        parser = HelpCenterParser(csv_path="/test/file.csv")
        assert "HelpCenterParser" in repr(parser)
        assert "/test/file.csv" in repr(parser)


class TestLoadArticles:
    """Tests for loading articles from CSV."""

    def test_load_articles_returns_list(self, tmp_path):
        """Load articles should return a list of Article objects."""
        # Create a temp CSV
        csv_path = tmp_path / "test.csv"
        csv_path.write_text('title,article_content\n"Test Title","<p>Test content</p>"')

        parser = HelpCenterParser(csv_path=str(csv_path))
        articles = parser.load_articles()

        assert isinstance(articles, list)
        assert len(articles) == 1
        assert isinstance(articles[0], Article)

    def test_load_articles_parses_content(self, tmp_path):
        """Articles should have parsed markdown content."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            'title,article_content\n"My Article","<h1>Title</h1><p>Body</p>"'
        )

        parser = HelpCenterParser(csv_path=str(csv_path))
        articles = parser.load_articles()

        assert articles[0].title == "My Article"
        assert "# Title" in articles[0].md_content
        assert "Body" in articles[0].md_content
        assert "<h1>" in articles[0].raw_html

    def test_load_articles_detects_features(self, tmp_path):
        """Articles should have detected content features."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "title,article_content\n"
            '"Integration Guide","<h1>Integration</h1><p>Step 1: Connect the API</p>"'
        )

        parser = HelpCenterParser(csv_path=str(csv_path))
        articles = parser.load_articles()

        assert articles[0].features.has_steps is True
        assert articles[0].features.is_integration is True

    def test_load_multiple_articles(self, tmp_path):
        """Multiple articles should be loaded correctly."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "title,article_content\n"
            '"Article 1","<p>Content 1</p>"\n'
            '"Article 2","<p>Content 2</p>"\n'
            '"Article 3","<p>Content 3</p>"'
        )

        parser = HelpCenterParser(csv_path=str(csv_path))
        articles = parser.load_articles()

        assert len(articles) == 3
        assert articles[0].title == "Article 1"
        assert articles[1].title == "Article 2"
        assert articles[2].title == "Article 3"
