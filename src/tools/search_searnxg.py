import requests
import logging
from typing import List
from src.core.config import Config
from src.tools.schemas import SearchResult
from langchain_community.utilities import SearxSearchWrapper

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Map standard "Intent" categories to SearXNG specific categories
CATEGORY_MAP = {
    "general": "general",
    "news": "news", 
    "science": "science",
    "it": "it",
    "files": "files",
    "images": "images",
    "videos": "videos"
}

def search_web(query: str, category: str = "general", time_range: str = "") -> list[SearchResult]:
    """
    Searches the web using SearXNG JSON API with robust filtering.
    
    Args:
        query: The search text.
        category: 'general', 'news', 'science', etc.
        time_range: 'day', 'week', 'month', 'year' (or empty string)
    """
    endpoint = f"{Config.SEARXNG_BASE_URL}/search"
    
    # Resolve the correct SearXNG category
    searx_category = CATEGORY_MAP.get(category, "general")
    
    # Build Params
    params = {
        "q": query,
        "format": "json",
        "categories": searx_category,
        "language": "auto"  # or Config.SEARCH_LANGUAGE if you have one
    }
    
    # Only add time_range if it is valid (SearXNG errors on empty strings sometimes)
    if time_range and time_range in ["day", "week", "month", "year"]:
        params["time_range"] = time_range

    try:
        logger.info(f"Searching web for: '{query}' | Cat: {searx_category} | Time: {time_range}")
        
        # 1. Execute Request
        response = requests.get(
            endpoint, 
            params=params, 
            timeout=Config.TIMEOUT
        )
        
        # 2. Check HTTP Status
        response.raise_for_status()
        
        # 3. Parse JSON
        data = response.json()
        
        # 4. Transform & Validate Data
        results = []
        raw_results = data.get("results", [])[:Config.MAX_RESULTS]
        
        for item in raw_results:
            content = item.get("content") or item.get("snippet") or ""
            
            # (Optional) JUNK FILTER: Skip social media profiles if looking for News
            url = item.get("url", "").lower()
            if category == "news" and any(x in url for x in ["instagram.com", "facebook.com", "twitter.com"]):
                continue

            result = SearchResult(
                title=item.get("title", "No Title"),
                url=item.get("url", ""),
                content=content
            )
            logger.info(f'appending web result: {result}')
            results.append(result)
            
        logger.info(f"✅ Found {len(results)} results.")
        return results

    except requests.exceptions.Timeout:
        logger.error("❌ Search timed out. SearXNG might be overloaded.")
        return []
    except requests.exceptions.ConnectionError:
        logger.error("❌ Could not connect to SearXNG. Is Docker running?")
        return []
    except Exception as e:
        logger.error(f"❌ Unexpected error during search: {e}")
        return []