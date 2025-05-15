""" Scrape docs pages & GitHub issues into text chunks for later processing. """
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import List, Dict, Any
import urllib3
from requests.exceptions import HTTPError
from urllib.parse import urljoin, urlparse
from tqdm import tqdm

# pull in .env vars (override any env you have)
load_dotenv(override=True)

# disable those annoying SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# config: GITHUB_TOKEN (for auth), DOCS_URL (where to start crawling), GITHUB_REPO (fallback repo)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
DOCS_URL = os.getenv("DOCS_URL")
GITHUB_REPO = os.getenv("GITHUB_REPO", "pingcap/tidb")
print(f"[Debug] Using DOCS_URL={DOCS_URL}")
if not DOCS_URL:
    raise ValueError("Please set DOCS_URL in your .env file")

def fetch_docs() -> List[Dict[str, Any]]:
    """ scrape docs pages recursively & split by H1/H2/H3 into chunks """
    visited = set()
    to_visit = [DOCS_URL.rstrip("/")]
    chunks: List[Dict[str, Any]] = []
    idx = 0

    # show progress as we scrape pages
    pbar = tqdm(desc="Scraping docs pages", unit="page")

    while to_visit:
        pbar.update(1)
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)
        try:
            resp = requests.get(url, verify=False)
            resp.raise_for_status()
        except Exception as e:
            print(f"[Warning] Could not fetch docs page {url}: {e}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        # slice content into chunks at headings (H1/H2/H3)
        for section in soup.find_all(['h1', 'h2', 'h3']):
            title = section.get_text(strip=True)
            content_parts = []
            for sibling in section.next_siblings:
                if getattr(sibling, "name", None) in ['h1', 'h2', 'h3']:
                    break
                if hasattr(sibling, "get_text"):
                    text = sibling.get_text(strip=True)
                    if text:
                        content_parts.append(text)
            if content_parts:
                text = f"{title}\n\n" + "\n\n".join(content_parts)
                chunks.append({"source": "docs", "id": str(idx), "chunk": text})
                idx += 1

        # queue up same-site links so we stay in docs
        for a in soup.find_all('a', href=True):
            href = a['href']
            # Resolve relative URLs
            full = urljoin(DOCS_URL, href)
            # Only crawl URLs under the same domain and path prefix
            parsed_base = urlparse(DOCS_URL)
            parsed_full = urlparse(full)
            if (parsed_full.netloc == parsed_base.netloc and
                parsed_full.path.startswith(parsed_base.path) and
                full not in visited):
                to_visit.append(full)

    pbar.close()
    return chunks

def fetch_github_issues(repo: str = None) -> List[Dict[str, Any]]:
    """ grab open GitHub issues from API """
    if repo is None:
        repo = GITHUB_REPO
    headers = {}
    # if you gave a token, use it
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    # GitHub API URL to fetch open issues (max 100)
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

# run it: fetch docs & issues, print counts
if __name__ == "__main__":
    docs = fetch_docs()
    issues = fetch_github_issues()
    print(f"Fetched {len(docs)} doc chunks and {len(issues)} GitHub issues.")