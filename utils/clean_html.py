from bs4 import BeautifulSoup
import re


def clean_html(text: str) -> str:
    """Clean HTML and normalize text."""
    if not text:
        return ""
    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove special characters but keep important ones
    text = re.sub(r"[^\w\s.,;:!?()-]", "", text)

    return text.strip()
