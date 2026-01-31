import logging
from typing import List, Optional
from ddgs import DDGS
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
    Searches the web using DuckDuckGo via 'ddgs' library.
    Matches the signature and behavior of the SearXNG implementation.
    
    Args:
        query: The search text.
        category: 'general', 'news', 'science', etc. (Used for logging/filtering context)
        time_range: 'day', 'week', 'month', 'year' (or empty string)
    """
    # 1. FAIL-SAFE: Category Validation
    # If agent guesses a category like 'finance', silently fallback to 'general'
    # to prevent the agent from getting stuck in an error loop.
    valid_categories = ["general", "news", "science", "it", "files", "images", "videos"]
    if category not in valid_categories:
        logger.warning(f"⚠️ Agent requested invalid category '{category}'. Defaulting to 'general'.")
        category = "general"

    # Resolve time limit
    timelimit = TIME_RANGE_MAP.get(time_range, None)

    try:
        logger.info(f"Searching web using DDG for: '{query}' | Cat: {category} | Time: {time_range}")
        
        results = []
        
        # Initialize DDGS context
        with DDGS() as ddgs:
            ddgs_gen = None
            
            if category == "news":
                ddgs_gen = ddgs.news(
                    query,
                    region='wt-wt',
                    safesearch='moderate',
                    timelimit=timelimit, # support d, w, m
                    max_results=Config.MAX_RESULTS
                )
            elif category == "images":
                ddgs_gen = ddgs.images(
                    query,
                    region='wt-wt',
                    safesearch='moderate',
                    timelimit=timelimit, # support Day, Week, Month
                    max_results=Config.MAX_RESULTS
                )
            elif category == "videos":
                 ddgs_gen = ddgs.videos(
                    query,
                    region='wt-wt',
                    safesearch='moderate',
                    timelimit=timelimit, # support w, m
                    max_results=Config.MAX_RESULTS
                )
            else:
                # General text search (Covers 'general', 'science', 'it', and fallbacks)
                ddgs_gen = ddgs.text(
                    query,
                    region='wt-wt',
                    safesearch='moderate',
                    timelimit=timelimit,
                    max_results=Config.MAX_RESULTS
                )
            
            # Process results if generator is not None
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
                    # Filter out noisy social media login pages often returned in news/web
                    if any(x in href.lower() for x in ["instagram.com/accounts/login", "facebook.com/login", "twitter.com/login"]):
                        continue

                    result = SearchResult(
                        title=title,
                        url=href,
                        content=body,
                        source="duckduckgo"
                    )
                    # Only log first few to avoid clutter
                    if len(results) < 3:
                        logger.info(f'appending web result: {title[:30]}...')
                    results.append(result)

        # 2. FAIL-SAFE: Empty Results
        # Return a specific object so the Agent knows it failed to find data
        if not results:
            logger.info("⚠️ No results found. returning empty state.")
            return [SearchResult(
                title="No Results Found",
                url="",
                content=f"The search engine returned 0 results for the query: '{query}'. Please try a different query or broader keywords.",
                source="system"
            )]

        logger.info(f"✅ Found {len(results)} results.")
        return results

    except Exception as e:
        logger.error(f"❌ Unexpected error during search: {e}")
        # Return a single error result instead of an empty list
        return [SearchResult(
            title="Error",
            url="http://error",
            content=f"Search tool failed with error: {str(e)}",
            source="error"
        )]