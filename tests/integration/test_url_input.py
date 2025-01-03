from document_to_podcast.preprocessing.data_cleaners import clean_html
from document_to_podcast.preprocessing import DATA_CLEANERS
import pytest
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

def test_url_content_cleaning():
    # Test with Mozilla blog
    url = "https://blog.mozilla.ai/introducing-blueprints-customizable-ai-workflows-for-developers/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    raw_text = soup.get_text()
    clean_text = DATA_CLEANERS[".html"](raw_text)
    
    # Verify cleaning maintains same quality as file upload
    assert len(clean_text) < len(raw_text)  # Should remove HTML
    assert "Mozilla" in clean_text  # Should preserve key content

def test_url_error_handling():
    with pytest.raises(RequestException):
        response = requests.get("https://nonexistent-url-that-should-fail.com")
        response.raise_for_status() 