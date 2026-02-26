"""
Unit tests for src/tools/search.py — covers both Tavily and Serper providers.
Patches at the class level so no real API calls are made.
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TAVILY_RESULTS = [
    {"title": "Article One", "url": "https://example.com/one", "content": "First result."},
    {"title": "Article Two", "url": "https://example.com/two", "content": "Second result."},
]

TAVILY_NEWS = [
    {"title": "News Item", "url": "https://news.example.com/story", "content": "Breaking news."},
]

SERPER_ORGANIC = {
    "organic": [
        {"title": "Serper One", "link": "https://example.com/s1", "snippet": "Serper snippet 1."},
        {"title": "Serper Two", "link": "https://example.com/s2", "snippet": "Serper snippet 2."},
    ]
}

SERPER_NEWS = {
    "news": [
        {"title": "Serper News", "link": "https://serper-news.com/story", "snippet": "Serper news snippet."},
    ]
}

JUNK_RESULTS = [
    {"title": "Real",          "url": "https://example.com/real",             "content": "OK"},
    {"title": "IG Login",      "url": "https://instagram.com/accounts/login", "content": ""},
    {"title": "FB Login",      "url": "https://facebook.com/login",           "content": ""},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_config(provider="tavily", tavily_key="t_key", serper_key="s_key", max_results=5):
    mock = MagicMock()
    mock.WEB_SEARCH_PROVIDER = provider
    mock.TAVILY_API_KEY = tavily_key
    mock.SERPER_API_KEY = serper_key
    mock.MAX_RESULTS = max_results
    return mock


# ---------------------------------------------------------------------------
# Tavily provider tests
# ---------------------------------------------------------------------------

class TestTavilyProvider:

    def _run(self, query, category="general", time_range="", results=TAVILY_RESULTS, tavily_key="t_key"):
        from src.tools.search import search_web
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = results
        with patch("src.tools.search.TavilySearchResults", return_value=mock_tool), \
             patch("src.tools.search.Config", _mock_config(provider="tavily", tavily_key=tavily_key)):
            return search_web(query, category=category, time_range=time_range)

    def test_organic_results_mapped_correctly(self):
        from src.tools.schemas import SearchResult
        res = self._run("test query")
        assert len(res) == 2
        assert all(isinstance(r, SearchResult) for r in res)
        assert res[0].title == "Article One"
        assert res[0].source == "tavily"

    def test_news_category(self):
        res = self._run("AI news", category="news", results=TAVILY_NEWS)
        assert len(res) == 1
        assert res[0].source == "tavily"

    def test_fallback_categories_still_return_results(self):
        for cat in ("science", "it", "images", "videos"):
            res = self._run("query", category=cat)
            assert len(res) > 0, f"Expected results for '{cat}'"

    def test_missing_key_returns_system_message(self):
        res = self._run("query", tavily_key="")
        assert res[0].source == "system"
        assert "TAVILY_API_KEY" in res[0].content

    def test_empty_results_returns_system_message(self):
        res = self._run("unfindable", results=[])
        assert res[0].source == "system"
        assert "0 results" in res[0].content

    def test_junk_filter(self):
        res = self._run("anything", results=JUNK_RESULTS)
        assert len(res) == 1
        assert res[0].url == "https://example.com/real"

    def test_api_exception_returns_error(self):
        from src.tools.search import search_web
        mock_tool = MagicMock()
        mock_tool.invoke.side_effect = RuntimeError("network failure")
        with patch("src.tools.search.TavilySearchResults", return_value=mock_tool), \
             patch("src.tools.search.Config", _mock_config(provider="tavily")):
            res = search_web("query")
        assert res[0].source == "error"
        assert "network failure" in res[0].content

    def test_invalid_category_defaults_to_general(self):
        res = self._run("stock market", category="finance")
        assert len(res) > 0

    def test_news_time_range_day_passes_days_param(self):
        from src.tools.search import search_web
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = TAVILY_NEWS
        with patch("src.tools.search.TavilySearchResults", return_value=mock_tool) as MockCls, \
             patch("src.tools.search.Config", _mock_config(provider="tavily")):
            search_web("latest news", category="news", time_range="day")
        assert MockCls.call_args.kwargs.get("days") == 1
        assert MockCls.call_args.kwargs.get("topic") == "news"


# ---------------------------------------------------------------------------
# Serper provider tests
# ---------------------------------------------------------------------------

class TestSerperProvider:

    def _run(self, query, category="general", time_range="", raw=SERPER_ORGANIC, serper_key="s_key"):
        from src.tools.search import search_web
        mock_wrapper = MagicMock()
        mock_wrapper.results.return_value = raw
        with patch("src.tools.search.GoogleSerperAPIWrapper", return_value=mock_wrapper), \
             patch("src.tools.search.Config", _mock_config(provider="serper", serper_key=serper_key)):
            return search_web(query, category=category, time_range=time_range)

    def test_organic_results_mapped_correctly(self):
        from src.tools.schemas import SearchResult
        res = self._run("test query")
        assert len(res) == 2
        assert all(isinstance(r, SearchResult) for r in res)
        assert res[0].title == "Serper One"
        assert res[0].source == "serper"

    def test_news_category(self):
        res = self._run("AI news", category="news", raw=SERPER_NEWS)
        assert len(res) == 1
        assert res[0].title == "Serper News"
        assert res[0].source == "serper"

    def test_missing_key_returns_system_message(self):
        res = self._run("query", serper_key="")
        assert res[0].source == "system"
        assert "SERPER_API_KEY" in res[0].content

    def test_empty_results_returns_system_message(self):
        res = self._run("unfindable", raw={"organic": []})
        assert res[0].source == "system"

    def test_api_exception_returns_error(self):
        from src.tools.search import search_web
        mock_wrapper = MagicMock()
        mock_wrapper.results.side_effect = RuntimeError("serper down")
        with patch("src.tools.search.GoogleSerperAPIWrapper", return_value=mock_wrapper), \
             patch("src.tools.search.Config", _mock_config(provider="serper")):
            res = search_web("query")
        assert res[0].source == "error"
        assert "serper down" in res[0].content


# ---------------------------------------------------------------------------
# Factory / provider selection tests
# ---------------------------------------------------------------------------

class TestProviderFactory:

    def test_unknown_provider_returns_error_result(self):
        from src.tools.search import search_web
        with patch("src.tools.search.Config", _mock_config(provider="bing")):
            res = search_web("query")
        assert res[0].source == "system"
