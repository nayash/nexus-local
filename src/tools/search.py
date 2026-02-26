"""
Web search module — supports multiple providers via a common interface.

Providers:
  - TavilySearchClient  : uses langchain_community TavilySearchResults
  - SerperSearchClient  : uses langchain_community GoogleSerperAPIWrapper

Active provider is controlled by Config.WEB_SEARCH_PROVIDER ("tavily" | "serper").
Switch in config.py or via the WEB_SEARCH_PROVIDER environment variable.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import GoogleSerperAPIWrapper

from src.core.config import Config
from src.tools.schemas import SearchResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VALID_CATEGORIES = {"general", "news", "science", "it", "files", "images", "videos"}

# Time range helpers
_TIME_RANGE_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
_TIME_RANGE_TBS  = {"day": "qdr:d", "week": "qdr:w", "month": "qdr:m", "year": "qdr:y"}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseSearchClient(ABC):
    """Common interface all search provider clients must implement."""

    @abstractmethod
    def search(self, query: str, category: str, time_range: str) -> List[SearchResult]:
        """
        Execute a search and return a list of SearchResult objects.

        Args:
            query:      Search text.
            category:   One of VALID_CATEGORIES.
            time_range: One of '', 'day', 'week', 'month', 'year'.
        """


# ---------------------------------------------------------------------------
# Tavily provider
# ---------------------------------------------------------------------------

class TavilySearchClient(BaseSearchClient):
    """
    Search client backed by Tavily via LangChain's TavilySearchResults tool.

    Requires TAVILY_API_KEY in the environment.
    Supported categories: general, news (others silently fall back to general).
    """

    _TOPIC_MAP = {"news": "news"}
    _FALLBACK_CATEGORIES = {"science", "it", "files", "images", "videos"}

    def __init__(self, api_key: str, max_results: int):
        self._api_key = api_key
        self._max_results = max_results

    def _build_tool(self, topic: str, days: Optional[int]) -> TavilySearchResults:
        kwargs = dict(
            tavily_api_key=self._api_key,
            max_results=self._max_results,
            topic=topic,
            include_answer=False,
        )
        if days is not None and topic == "news":
            kwargs["days"] = days
        return TavilySearchResults(**kwargs)

    def search(self, query: str, category: str, time_range: str) -> List[SearchResult]:
        if category in self._FALLBACK_CATEGORIES:
            logger.warning(f"⚠️ Tavily: category '{category}' unsupported — falling back to general.")

        topic = self._TOPIC_MAP.get(category, "general")
        days  = _TIME_RANGE_DAYS.get(time_range, None)
        tool  = self._build_tool(topic=topic, days=days)

        # Returns: [{"title": ..., "url": ..., "content": ...}, ...]
        raw: list = tool.invoke(query)

        return [
            SearchResult(
                title=item.get("title", "No Title"),
                url=item.get("url", ""),
                content=item.get("content", ""),
                source="tavily",
            )
            for item in raw
        ]


# ---------------------------------------------------------------------------
# Serper provider
# ---------------------------------------------------------------------------

class SerperSearchClient(BaseSearchClient):
    """
    Search client backed by Serper.dev via LangChain's GoogleSerperAPIWrapper.

    Requires SERPER_API_KEY in the environment.
    Supported categories: general, news, images (others fall back to general).
    """

    _TYPE_MAP = {"news": "news", "images": "images"}
    _FALLBACK_CATEGORIES = {"science", "it", "files", "videos"}

    def __init__(self, api_key: str, max_results: int):
        self._api_key = api_key
        self._max_results = max_results

    def _build_wrapper(self, search_type: str, tbs: Optional[str]) -> GoogleSerperAPIWrapper:
        kwargs = dict(
            serper_api_key=self._api_key,
            k=self._max_results,
            type=search_type,
        )
        if tbs:
            kwargs["tbs"] = tbs
        return GoogleSerperAPIWrapper(**kwargs)

    def search(self, query: str, category: str, time_range: str) -> List[SearchResult]:
        if category in self._FALLBACK_CATEGORIES:
            logger.warning(f"⚠️ Serper: category '{category}' unsupported — falling back to general.")

        search_type = self._TYPE_MAP.get(category, "search")
        tbs         = _TIME_RANGE_TBS.get(time_range, None)
        wrapper     = self._build_wrapper(search_type=search_type, tbs=tbs)

        # Returns a dict with keys: "organic", "news", "images", ...
        raw: dict = wrapper.results(query)

        results: List[SearchResult] = []

        if search_type == "news":
            for item in raw.get("news", []):
                results.append(SearchResult(
                    title=item.get("title", "No Title"),
                    url=item.get("link", ""),
                    content=item.get("snippet", ""),
                    source="serper",
                ))
        elif search_type == "images":
            for item in raw.get("images", []):
                results.append(SearchResult(
                    title=item.get("title", "No Title"),
                    url=item.get("imageUrl", "") or item.get("link", ""),
                    content=f"Source: {item.get('source', '')} | {item.get('imageWidth', '')}x{item.get('imageHeight', '')}",
                    source="serper",
                ))
        else:
            for item in raw.get("organic", []):
                results.append(SearchResult(
                    title=item.get("title", "No Title"),
                    url=item.get("link", ""),
                    content=item.get("snippet", ""),
                    source="serper",
                ))

        return results


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def get_search_client() -> Optional[BaseSearchClient]:
    """
    Instantiate the configured search provider.

    Returns None (with a logged error) if the required API key is missing.
    Provider is selected via Config.WEB_SEARCH_PROVIDER ("tavily" | "serper").
    """
    provider = Config.WEB_SEARCH_PROVIDER.lower()

    if provider == "tavily":
        api_key = Config.TAVILY_API_KEY
        if not api_key:
            logger.error("❌ TAVILY_API_KEY is not set. Web search is disabled.")
            return None
        return TavilySearchClient(api_key=api_key, max_results=Config.MAX_RESULTS)

    if provider == "serper":
        api_key = Config.SERPER_API_KEY
        if not api_key:
            logger.error("❌ SERPER_API_KEY is not set. Web search is disabled.")
            return None
        return SerperSearchClient(api_key=api_key, max_results=Config.MAX_RESULTS)

    logger.error(f"❌ Unknown WEB_SEARCH_PROVIDER '{provider}'. Must be 'tavily' or 'serper'.")
    return None


# ---------------------------------------------------------------------------
# Public entry point (unchanged signature)
# ---------------------------------------------------------------------------

def search_web(query: str, category: str = "general", time_range: str = "") -> List[SearchResult]:
    """
    Search the web using the configured provider (Tavily or Serper).

    Args:
        query:      The search text.
        category:   One of 'general', 'news', 'science', 'it', 'files', 'images', 'videos'.
        time_range: One of '' (anytime), 'day', 'week', 'month', 'year'.

    Returns:
        A list of SearchResult objects — always non-empty (system message on failure).
    """
    # --- Category validation ---
    if category not in VALID_CATEGORIES:
        logger.warning(f"⚠️ Invalid category '{category}'. Defaulting to 'general'.")
        category = "general"

    # --- Resolve provider ---
    client = get_search_client()
    if client is None:
        provider = Config.WEB_SEARCH_PROVIDER
        key_name = "TAVILY_API_KEY" if provider == "tavily" else "SERPER_API_KEY"
        return [SearchResult(
            title="Web Search Unavailable",
            url="",
            content=f"Web search is disabled because {key_name} is not configured.",
            source="system",
        )]

    try:
        logger.info(
            f"Searching via {Config.WEB_SEARCH_PROVIDER}: '{query}' | "
            f"Cat: {category} | Time: {time_range}"
        )
        results = client.search(query=query, category=category, time_range=time_range)

        # --- Junk filter ---
        results = [
            r for r in results
            if not any(
                x in r.url.lower()
                for x in ["instagram.com/accounts/login", "facebook.com/login", "twitter.com/login"]
            )
        ]

        for r in results[:3]:
            logger.info(f"  ↳ {r.title[:50]}...")

        if not results:
            logger.warning("⚠️ No results found.")
            return [SearchResult(
                title="No Results Found",
                url="",
                content=f"The search engine returned 0 results for: '{query}'. Try different keywords.",
                source="system",
            )]

        logger.info(f"✅ {len(results)} results returned.")
        return results

    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return [SearchResult(
            title="Error",
            url="http://error",
            content=f"Search tool failed with error: {str(e)}",
            source="error",
        )]