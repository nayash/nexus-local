import logging
from typing import List, Optional
from duckduckgo_search import DDGS
from src.core.config import Config
from src.tools.schemas import SearchResult

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Map standard "Intent" categories to DuckDuckGo specific behavior if needed
# For now, we primarily just log them as DDG is a general engine.
CATEGORY_MAP = {
    "general": "general",
    "news": "news", 
    "science": "science",
    "it": "it",
    "files": "files",
    "images": "images",
    "videos": "videos"
}

# Map time_range to DDGS timelimit codes
TIME_RANGE_MAP = {
    "day": "d",
    "week": "w",
    "month": "m",
    "year": "y"
}

def search_web(query: str, category: str = "general", time_range: str = "") -> List[SearchResult]:
    """
    Searches the web using DuckDuckGo via duckduckgo_search library.
    Matches the signature and behavior of the SearXNG implementation.
    
    Args:
        query: The search text.
        category: 'general', 'news', 'science', etc. (Used for logging/filtering context)
        time_range: 'day', 'week', 'month', 'year' (or empty string)
    """
    # Resolve time limit
    timelimit = TIME_RANGE_MAP.get(time_range, None)

    try:
        logger.info(f"Searching web using DDG for: '{query}' | Cat: {category} | Time: {time_range}")
        
        results = []
        
        # Initialize DDGS context
        with DDGS() as ddgs:
            if category == "news":
                ddgs_gen = ddgs.news(
                    keywords=query,
                    region='wt-wt',
                    safesearch='moderate',
                    timelimit=timelimit, # support d, w, m
                    max_results=Config.MAX_RESULTS
                )
            elif category == "images":
                ddgs_gen = ddgs.images(
                    keywords=query,
                    region='wt-wt',
                    safesearch='moderate',
                    timelimit=timelimit, # support Day, Week, Month
                    max_results=Config.MAX_RESULTS
                )
            elif category == "videos":
                 ddgs_gen = ddgs.videos(
                    keywords=query,
                    region='wt-wt',
                    safesearch='moderate',
                    timelimit=timelimit, # support w, m
                    max_results=Config.MAX_RESULTS
                )
            else:
                # General text search
                ddgs_gen = ddgs.text(
                    keywords=query,
                    region='wt-wt',
                    safesearch='moderate',
                    timelimit=timelimit,
                    max_results=Config.MAX_RESULTS
                )
            
            if ddgs_gen:
                for item in ddgs_gen:
                    # Normalize fields based on result type
                    title = item.get("title", "No Title")
                    
                    if category == "images":
                        href = item.get("image", "") or item.get("url", "")
                        body = f"Source: {item.get('source','')} | Width: {item.get('width','')} | Height: {item.get('height','')}"
                    elif category == "videos":
                         href = item.get("content", "") # video url usually here or 'embed_url'
                         if not href: href = item.get("embed_url", "")
                         body = item.get("description", "")
                    elif category == "news":
                        href = item.get("url", "")
                        body = item.get("body", "") or item.get("excerpt", "")
                    else:
                        href = item.get("href", "")
                        body = item.get("body", "")
                    
                    # (Optional) JUNK FILTER implementation
                    if category == "news" and any(x in href.lower() for x in ["instagram.com", "facebook.com", "twitter.com"]):
                        continue

                    result = SearchResult(
                        title=title,
                        url=href,
                        content=body,
                        source="duckduckgo"
                    )
                    logger.info(f'appending web result: {title[:30]}...')
                    results.append(result)

        logger.info(f"✅ Found {len(results)} results.")
        return results

    except Exception as e:
        logger.error(f"❌ Unexpected error during search: {e}")
        # Return a single error result instead of an empty list
        return [SearchResult(
            title="Error",
            url="http://error",
            content=f"Search failed: {str(e)}",
            source="error"
        )]
