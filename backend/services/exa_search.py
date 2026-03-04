import os
from exa_py import Exa


def search_people(query: str, max_results: int = 10) -> list:
    """Search for people/founders using Exa API.

    Returns list of {title, url, highlights} dicts.
    """
    exa = Exa(os.getenv("EXA_API_KEY"))
    exa_query = f"Founders of {query} startups that were founded within the last 2 years"
    response = exa.search(
        exa_query,
        category="people",
        type="deep",
        num_results=max_results,
        contents={"highlights": {"max_characters": 4000}},
    )
    results = []
    for r in response.results:
        highlights = []
        if hasattr(r, "highlights") and r.highlights:
            highlights = r.highlights
        results.append({
            "title": getattr(r, "title", "") or "",
            "url": getattr(r, "url", "") or "",
            "highlights": highlights,
        })
    return results
