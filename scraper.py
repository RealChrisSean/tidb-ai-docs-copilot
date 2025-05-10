import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import List, Dict, Any
import urllib3
from requests.exceptions import HTTPError

# Load variables from .env in project root
load_dotenv()

# Disable SSL warnings (optional)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Read configuration from environment
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
DOCS_URL = os.getenv("DOCS_URL")
print(f"[Debug] Using DOCS_URL={DOCS_URL}")
if not DOCS_URL:
    raise ValueError("Please set DOCS_URL in your .env file")

def fetch_docs() -> List[Dict[str, Any]]:
    """
    Fetch and parse documentation pages into text chunks.
    Returns a list of dicts with keys: source, id, chunk.
    """
    try:
        response = requests.get(DOCS_URL, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        chunks: List[Dict[str, Any]] = []
        for idx, section in enumerate(soup.find_all(['h1', 'h2', 'h3'])):
            title = section.get_text(strip=True)
            content_parts = []
            for sibling in section.next_siblings:
                if getattr(sibling, "name", None) and sibling.name in ['h1', 'h2', 'h3']:
                    break
                if hasattr(sibling, "get_text"):
                    text = sibling.get_text(strip=True)
                    if text:
                        content_parts.append(text)
            if content_parts:
                text = f"{title}\n\n" + "\n\n".join(content_parts)
                chunks.append({"source": "docs", "id": str(idx), "chunk": text})
        return chunks
    except HTTPError as e:
        print(f"[Warning] Could not fetch docs from {DOCS_URL}: {e}")
        return []
    except Exception as e:
        print(f"[Warning] Error fetching docs: {e}")
        return []

def fetch_github_issues(repo: str = "tidb/tidb") -> List[Dict[str, Any]]:
    """
    Fetch open GitHub issues for the given repo.
    Returns a list of dicts with keys: source, id, chunk.
    """
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    url = f"https://api.github.com/repos/{repo}/issues?state=open&per_page=100"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        issues: List[Dict[str, Any]] = []
        for issue in response.json():
            number = issue.get("number")
            title = issue.get("title", "")
            body = issue.get("body", "")
            text = f"{title}\n\n{body}"
            issues.append({"source": "github", "id": str(number), "chunk": text})
        return issues
    except HTTPError as e:
        print(f"[Warning] Could not fetch GitHub issues from {url}: {e}")
        return []
    except Exception as e:
        print(f"[Warning] Error fetching GitHub issues: {e}")
        return []

if __name__ == "__main__":
    docs = fetch_docs()
    issues = fetch_github_issues()
    print(f"Fetched {len(docs)} doc chunks and {len(issues)} GitHub issues.")