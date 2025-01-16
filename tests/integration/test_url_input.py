from document_to_podcast.preprocessing import DATA_CLEANERS
import pytest
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException


def test_url_content_cleaning():
    """Test basic URL content fetching and cleaning."""
    url = "https://blog.mozilla.ai/introducing-blueprints-customizable-ai-workflows-for-developers/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    raw_text = soup.get_text()
    clean_text = DATA_CLEANERS[".html"](raw_text)

    # Verify cleaning maintains same quality as file upload
    assert len(clean_text) < len(raw_text)  # Should remove HTML
    assert "Mozilla" in clean_text  # Should preserve key content


def test_url_error_handling():
    """Test handling of network errors."""
    with pytest.raises(RequestException):
        response = requests.get("https://nonexistent-url-that-should-fail.com")
        response.raise_for_status()


def test_url_content_quality():
    """Test that cleaned URL content maintains expected quality."""
    url = "https://blog.mozilla.org/en/mozilla/introducing-mozilla-ai-investing-in-trustworthy-ai/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    raw_text = soup.get_text()
    clean_text = DATA_CLEANERS[".html"](raw_text)

    # Test content quality
    assert "mozilla" in clean_text.lower()  # Key terms preserved
    assert "ai" in clean_text.lower()  # Case-insensitive content check
    assert "<html>" not in clean_text  # HTML tags removed
    assert "utm_source" not in clean_text  # Marketing parameters removed


def test_url_content_size_limits():
    """Test handling of different content sizes."""
    url = "https://www.mozilla.org/en-US/about/manifesto/"  # Substantial page
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    raw_text = soup.get_text()
    clean_text = DATA_CLEANERS[".html"](raw_text)

    # Size checks
    assert len(clean_text) > 100  # Not too small
    assert len(clean_text) < len(raw_text)  # Smaller than raw
    assert len(clean_text.split()) > 50  # Has substantial content
