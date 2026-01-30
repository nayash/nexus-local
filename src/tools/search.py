from typing import List, Optional
from dataclasses import dataclass
from duckduckgo_search import DDGS
import sys

@dataclass
class SearchResult:
    title: str
    href: str
    body: str
    error: Optional[str] = None

def search_web(query: str, max_results: int = 5) -> List[SearchResult]:
    """
    Perform a web search using DuckDuckGo.
    
    Args:
        query: The search query.
        max_results: Max number of results to return.
        
    Returns:
        List[SearchResult]: A list of search results.
        If an error occurs, returns a single SearchResult with the error message.
    """
    try:
        results = []
        with DDGS() as ddgs:
            # ddgs.text() returns an iterator of dicts {'title':..., 'href':..., 'body':...}
            ddgs_gen = ddgs.text(query, max_results=max_results)
            if ddgs_gen:
                for r in ddgs_gen:
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        href=r.get("href", ""),
                        body=r.get("body", "")
                    ))
        
        return results

    except Exception as e:
        # Return a single error result instead of raising
        return [SearchResult(
            title="Error",
            href="",
            body="",
            error=f"Search failed: {str(e)}"
        )]
