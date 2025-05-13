# scrape.py
from bs4 import BeautifulSoup
import requests
import json

urls = ["https://job-compass.bunkid.online/", "https://job-compass.bunkid.online/terms-of-service"]  # Replace with your URLs
website_chunks = []
for url in urls:
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join(p.get_text() for p in soup.find_all("p"))
        website_chunks.append({"content": text, "url": url})
    except Exception as e:
        print(f"Error scraping {url}: {e}")

with open("website_chunks.json", "w") as f:
    json.dump(website_chunks, f)