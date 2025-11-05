import requests
import csv
import re
import html
import os
from dotenv import load_dotenv

load_dotenv()

def build_query(keywords, role, company):
    """
    Build a Google search query for LinkedIn profiles.
    Only includes non-empty fields in the query.
    """
    parts = []
    
    if keywords:
        parts.append(f'"{keywords}"')
    if role:
        parts.append(f'"{role}"')
    if company:
        parts.append(f'"{company}"')
    
    query_parts = ' '.join(parts)
    return f'site:linkedin.com/in/ {query_parts}'

def parse_title(title):
    """
    Attempt to split title into name, job title, and company.
    Example: "Peter Johnson, DBIA - Sr. Preconstruction Manager at Faber Construction"
    """
    name, job_title, company = None, None, None
    if not title:
        return None, None, None

    # Split at first ' - '
    parts = title.split(" - ", 1)
    if len(parts) == 2:
        name = parts[0].strip()
        right_part = parts[1]
    else:
        right_part = title

    # Try to extract job title and company
    match = re.search(r"(?P<role>.+?)\s+at\s+(?P<company>.+)", right_part)
    if match:
        job_title = match.group("role").strip()
        company = match.group("company").strip().split("|")[0]  # clean trailing LinkedIn or extra text
    else:
        job_title = right_part.strip()

    return name, job_title, company

def clean_text(text):
    """Unescape HTML entities and remove newlines/extra spaces."""
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def google_search(query):
    API_KEY = os.getenv("GOOGLE_API_KEY")
    CX = os.getenv("GOOGLE_CX")
    if not API_KEY or not CX:
        raise ValueError("GOOGLE_API_KEY and GOOGLE_CX must be set")
    url = "https://www.googleapis.com/customsearch/v1"
    
    all_results = []
    for start in range(1, 100, 10):
        params = {
            "key": API_KEY,
            "cx": CX,
            "q": query,
            "start": start
        }
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()

        # Check for API errors
        if "error" in data:
            error_msg = data["error"].get("message", "Unknown error")
            raise ValueError(f"Google API error: {error_msg}")

        items = data.get("items", [])
        if not items:
            break

        for item in items:
            pagemap = item.get("pagemap", {})
            metatags = pagemap.get("metatags", [{}])[0]
            
            # Try multiple sources for title - Google API provides title at item level
            title = item.get("title") or metatags.get("og:title") or ""
            title = clean_text(title)
            title = title.replace(" | LinkedIn", "")

            name, role, company = parse_title(title)

            result = {
                "name": name,
                "job_title": role,
                "company": company,
                "link": item.get("link"),
                "title": clean_text(metatags.get("og:title") or item.get("title", "")),
                "description": clean_text(metatags.get("og:description") or item.get("snippet", "")),
            }

            all_results.append(result)

    return all_results