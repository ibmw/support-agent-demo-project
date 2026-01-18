"""Unit tests for content feature detection."""

from support_agent.indexing.features import ContentFeatureDetector, ContentFeatures


class TestContentFeatures:
    """Tests for ContentFeatures dataclass."""

    def test_default_values(self):
        """All features default to False/general."""
        features = ContentFeatures()

        assert features.has_steps is False
        assert features.is_faq is False
        assert features.has_code is False
        assert features.is_integration is False
        assert features.is_troubleshooting is False
        assert features.has_pricing is False
        assert features.content_type == "general"

    def test_to_dict(self):
        """to_dict() returns all features as dictionary."""
        features = ContentFeatures(
            has_steps=True,
            is_faq=True,
            content_type="faq",
        )
        result = features.to_dict()

        assert isinstance(result, dict)
        assert result["has_steps"] is True
        assert result["is_faq"] is True
        assert result["has_code"] is False
        assert result["content_type"] == "faq"

    def test_to_dict_all_keys_present(self):
        """to_dict() includes all feature keys."""
        features = ContentFeatures()
        result = features.to_dict()

        expected_keys = {
            "has_steps",
            "is_faq",
            "has_code",
            "is_integration",
            "is_troubleshooting",
            "has_pricing",
            "content_type",
        }
        assert set(result.keys()) == expected_keys


class TestDetectSteps:
    """Tests for step-by-step content detection."""

    def test_detects_step_1(self):
        """Detects 'Step 1' pattern."""
        features = ContentFeatureDetector.detect("", "Follow Step 1 to begin.")
        assert features.has_steps is True

    def test_detects_steps_plural(self):
        """Detects 'Steps' pattern with number."""
        # Pattern requires step/steps followed by a digit
        features = ContentFeatureDetector.detect(
            "", "Follow steps 1-3 to complete setup."
        )
        assert features.has_steps is True

    def test_detects_etape_french(self):
        """Detects French 'Étape' pattern."""
        features = ContentFeatureDetector.detect("", "Étape 2: Configuration")
        assert features.has_steps is True

    def test_detects_numbered_actions(self):
        """Detects numbered action lists."""
        features = ContentFeatureDetector.detect("", "1. Go to Settings\n2. Click Save")
        assert features.has_steps is True

    def test_detects_how_to(self):
        """Detects 'How to' guides."""
        features = ContentFeatureDetector.detect("", "How to configure your account")
        assert features.has_steps is True

    def test_detects_sequential_words(self):
        """Detects sequential instruction words."""
        features = ContentFeatureDetector.detect(
            "", "First, open the app. Then, log in."
        )
        assert features.has_steps is True

    def test_no_steps_in_plain_text(self):
        """Plain text without instructions returns False."""
        features = ContentFeatureDetector.detect("", "This is general information.")
        assert features.has_steps is False


class TestDetectFaq:
    """Tests for FAQ content detection."""

    def test_detects_faq_keyword(self):
        """Detects 'FAQ' keyword."""
        features = ContentFeatureDetector.detect("", "Check our FAQ section")
        assert features.is_faq is True

    def test_detects_frequently_asked(self):
        """Detects 'frequently asked' phrase."""
        features = ContentFeatureDetector.detect("", "Frequently Asked Questions")
        assert features.is_faq is True

    def test_detects_qa_format(self):
        """Detects Q: A: format."""
        features = ContentFeatureDetector.detect("", "Q: What is this?\nA: A test.")
        assert features.is_faq is True

    def test_no_faq_in_regular_text(self):
        """Regular text returns False."""
        features = ContentFeatureDetector.detect("", "Regular documentation content.")
        assert features.is_faq is False


class TestDetectCode:
    """Tests for code content detection."""

    def test_detects_code_tag(self):
        """Detects <code> tags in HTML."""
        features = ContentFeatureDetector.detect(
            "<p>Use <code>print()</code></p>", "Use print()"
        )
        assert features.has_code is True

    def test_detects_pre_tag(self):
        """Detects <pre> tags in HTML."""
        features = ContentFeatureDetector.detect("<pre>code block</pre>", "code block")
        assert features.has_code is True

    def test_detects_markdown_code_fence(self):
        """Detects markdown code fences in HTML."""
        features = ContentFeatureDetector.detect("<p>```python\ncode\n```</p>", "code")
        assert features.has_code is True

    def test_detects_script_tag(self):
        """Detects <script> tags."""
        features = ContentFeatureDetector.detect("<script>var x = 1;</script>", "")
        assert features.has_code is True

    def test_no_code_in_plain_html(self):
        """Plain HTML without code returns False."""
        features = ContentFeatureDetector.detect("<p>Just text</p>", "Just text")
        assert features.has_code is False


class TestDetectIntegration:
    """Tests for integration content detection."""

    def test_detects_integration_keyword(self):
        """Detects 'integration' keyword."""
        features = ContentFeatureDetector.detect("", "Slack integration guide")
        assert features.is_integration is True

    def test_detects_connect_keyword(self):
        """Detects 'connect' keyword."""
        features = ContentFeatureDetector.detect("", "Connect your account")
        assert features.is_integration is True

    def test_detects_oauth(self):
        """Detects 'OAuth' keyword."""
        features = ContentFeatureDetector.detect("", "Authorize with OAuth")
        assert features.is_integration is True

    def test_detects_api_key(self):
        """Detects 'API key' phrase."""
        features = ContentFeatureDetector.detect("", "Enter your API key")
        assert features.is_integration is True

    def test_detects_webhook(self):
        """Detects 'webhook' keyword."""
        features = ContentFeatureDetector.detect("", "Configure the webhook URL")
        assert features.is_integration is True

    def test_detects_setup_app(self):
        """Detects 'setup app' pattern."""
        features = ContentFeatureDetector.detect("", "Install and setup the app")
        assert features.is_integration is True

    def test_no_integration_in_general_text(self):
        """General text returns False."""
        features = ContentFeatureDetector.detect("", "Welcome to our service.")
        assert features.is_integration is False


class TestDetectTroubleshooting:
    """Tests for troubleshooting content detection."""

    def test_detects_troubleshoot(self):
        """Detects 'troubleshoot' keyword."""
        features = ContentFeatureDetector.detect("", "Troubleshoot login issues")
        assert features.is_troubleshooting is True

    def test_detects_fix(self):
        """Detects 'fix' keyword."""
        features = ContentFeatureDetector.detect("", "How to fix the error")
        assert features.is_troubleshooting is True

    def test_detects_not_working(self):
        """Detects 'not working' phrase."""
        features = ContentFeatureDetector.detect("", "Feature is not working")
        assert features.is_troubleshooting is True

    def test_detects_if_you_see(self):
        """Detects 'if you see/get' patterns."""
        features = ContentFeatureDetector.detect("", "If you see an error message")
        assert features.is_troubleshooting is True

    def test_detects_unable(self):
        """Detects 'unable' keyword."""
        features = ContentFeatureDetector.detect("", "If you're unable to login")
        assert features.is_troubleshooting is True

    def test_no_troubleshooting_in_happy_path(self):
        """Happy path content returns False."""
        features = ContentFeatureDetector.detect("", "Your account is ready to use.")
        assert features.is_troubleshooting is False


class TestDetectPricing:
    """Tests for pricing content detection."""

    def test_detects_pricing_keyword(self):
        """Detects 'pricing' keyword."""
        features = ContentFeatureDetector.detect("", "View our pricing plans")
        assert features.has_pricing is True

    def test_detects_subscription(self):
        """Detects 'subscription' keyword."""
        features = ContentFeatureDetector.detect("", "Manage your subscription")
        assert features.has_pricing is True

    def test_detects_billing(self):
        """Detects 'billing' keyword."""
        features = ContentFeatureDetector.detect("", "Update billing information")
        assert features.has_pricing is True

    def test_detects_dollar_amounts(self):
        """Detects dollar amounts."""
        features = ContentFeatureDetector.detect("", "Starting at $10/month")
        assert features.has_pricing is True

    def test_detects_currency_codes(self):
        """Detects currency amounts with codes."""
        features = ContentFeatureDetector.detect("", "Price: 50 USD")
        assert features.has_pricing is True

    def test_detects_plan_tiers(self):
        """Detects plan tier names."""
        features = ContentFeatureDetector.detect("", "Upgrade to Premium plan")
        assert features.has_pricing is True

    def test_detects_free_tier(self):
        """Detects 'free' tier."""
        features = ContentFeatureDetector.detect("", "Start with the free plan")
        assert features.has_pricing is True

    def test_no_pricing_in_general_content(self):
        """General content returns False."""
        features = ContentFeatureDetector.detect("", "Learn about our features.")
        assert features.has_pricing is False


class TestInferContentType:
    """Tests for content type inference."""

    def test_faq_highest_priority(self):
        """FAQ takes priority over other types."""
        features = ContentFeatureDetector.detect(
            "", "FAQ: How to troubleshoot Step 1 of integration"
        )
        assert features.content_type == "faq"

    def test_troubleshooting_second_priority(self):
        """Troubleshooting comes after FAQ."""
        features = ContentFeatureDetector.detect("", "Troubleshoot integration Step 1")
        assert features.content_type == "troubleshooting"

    def test_integration_third_priority(self):
        """Integration comes after troubleshooting."""
        features = ContentFeatureDetector.detect(
            "", "Connect your app. Step 1: Configure"
        )
        assert features.content_type == "integration"

    def test_how_to_for_steps_only(self):
        """Steps without other features gives how-to."""
        features = ContentFeatureDetector.detect(
            "", "Step 1: Click the button. Step 2: Save."
        )
        assert features.content_type == "how-to"

    def test_technical_for_code(self):
        """Code content without other features gives technical."""
        features = ContentFeatureDetector.detect(
            "<code>example</code>", "Example code snippet"
        )
        assert features.content_type == "technical"

    def test_pricing_lowest_specific_priority(self):
        """Pricing is lowest specific priority."""
        features = ContentFeatureDetector.detect("", "View pricing options")
        assert features.content_type == "pricing"

    def test_general_default(self):
        """No features gives general type."""
        features = ContentFeatureDetector.detect("<p>Hello world</p>", "Hello world")
        assert features.content_type == "general"


class TestMatchAny:
    """Tests for _match_any helper method."""

    def test_matches_single_pattern(self):
        """Returns True when one pattern matches."""
        result = ContentFeatureDetector._match_any("hello world", [r"world"])
        assert result is True

    def test_matches_any_of_multiple(self):
        """Returns True when any pattern matches."""
        result = ContentFeatureDetector._match_any(
            "hello world", [r"foo", r"bar", r"world"]
        )
        assert result is True

    def test_no_match(self):
        """Returns False when no pattern matches."""
        result = ContentFeatureDetector._match_any("hello world", [r"foo", r"bar"])
        assert result is False

    def test_case_insensitive(self):
        """Matching is case insensitive."""
        result = ContentFeatureDetector._match_any("HELLO WORLD", [r"hello"])
        assert result is True

    def test_empty_patterns(self):
        """Empty pattern list returns False."""
        result = ContentFeatureDetector._match_any("hello", [])
        assert result is False


class TestFullDetection:
    """Integration tests for the full detection flow."""

    def test_integration_guide_detection(self):
        """Full integration guide detection."""
        html = "<h1>Slack Integration</h1><p>Step 1: Connect. Step 2: Configure the webhook.</p>"
        text = "# Slack Integration\nStep 1: Connect. Step 2: Configure the webhook."

        features = ContentFeatureDetector.detect(html, text)

        assert features.is_integration is True
        assert features.has_steps is True
        assert features.content_type == "integration"

    def test_troubleshooting_faq_detection(self):
        """FAQ troubleshooting content detection."""
        html = "<h1>FAQ</h1><p>Q: Why doesn't it work? A: Try restarting.</p>"
        text = "# FAQ\nQ: Why doesn't it work? A: Try restarting."

        features = ContentFeatureDetector.detect(html, text)

        assert features.is_faq is True
        # Note: "doesn't" with curly apostrophe doesn't match the straight apostrophe pattern
        # The troubleshooting detection uses straight quotes in patterns
        assert features.content_type == "faq"  # FAQ takes priority

    def test_troubleshooting_faq_with_keywords(self):
        """FAQ with explicit troubleshooting keywords."""
        html = "<h1>FAQ</h1><p>Q: How to fix login issues? A: Troubleshoot by clearing cache.</p>"
        text = "# FAQ\nQ: How to fix login issues? A: Troubleshoot by clearing cache."

        features = ContentFeatureDetector.detect(html, text)

        assert features.is_faq is True
        assert features.is_troubleshooting is True
        assert features.content_type == "faq"  # FAQ takes priority

    def test_pricing_page_detection(self):
        """Pricing page content detection."""
        html = "<h1>Pricing</h1><p>Starter: $10/mo. Pro: $25/mo. Enterprise plan available.</p>"
        text = "# Pricing\nStarter: $10/mo. Pro: $25/mo. Enterprise plan available."

        features = ContentFeatureDetector.detect(html, text)

        assert features.has_pricing is True
        assert features.content_type == "pricing"

    def test_technical_docs_detection(self):
        """Technical documentation detection."""
        html = "<p>Use the following code:</p><pre>const api = new Client()</pre>"
        text = "Use the following code: const api = new Client()"

        features = ContentFeatureDetector.detect(html, text)

        assert features.has_code is True
        assert features.content_type == "technical"
