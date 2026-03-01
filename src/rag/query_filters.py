import os
import re
from datetime import date, timedelta
from functools import lru_cache
from typing import Optional, Tuple

from langchain_classic.chains.query_constructor.base import AttributeInfo, load_query_constructor_runnable
from langchain_core.structured_query import Comparator, Comparison, Operation, Operator
from langchain_ollama import ChatOllama

from src.core.config import Config
from src.core.user_settings import get_setting
from src.rag.translators import LanceDBTranslator


_FILENAME_PATTERN = re.compile(r"\b([\w\-. ]+\.(pdf|txt|md|csv|sh|py|docx|html|htm|jpg|jpeg|png|log))\b", re.IGNORECASE)


def _today() -> date:
    return date.today()


def _normalize_relative_dates(query: str) -> str:
    today = _today()
    replacements = {
        r"\byesterday\b": (today - timedelta(days=1)).isoformat(),
        r"\btoday\b": today.isoformat(),
    }
    normalized = query
    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


@lru_cache(maxsize=1)
def _get_filter_llm():
    return ChatOllama(
        model=get_setting("model_name", "llama3.1"),
        temperature=0,
        base_url=Config.OLLAMA_BASE_URL,
    )


@lru_cache(maxsize=1)
def _get_query_constructor():
    attribute_info = [
        AttributeInfo(name="file_name", description="The source filename including extension.", type="string"),
        AttributeInfo(name="source_path", description="Absolute source path for the ingested file.", type="string"),
        AttributeInfo(name="source_type", description="The file type such as pdf, docx, csv, txt, image, html.", type="string"),
        AttributeInfo(name="document_kind", description="High-level content kind such as book, log, data, image, document, code.", type="string"),
        AttributeInfo(name="author", description="Document or book author when available.", type="string"),
        AttributeInfo(name="title", description="Document title when available.", type="string"),
        AttributeInfo(name="owner", description="Filesystem owner username.", type="string"),
        AttributeInfo(name="source_mtime_date", description="Last modified date in ISO format YYYY-MM-DD.", type="string"),
        AttributeInfo(name="workspace_id", description="Logical workspace or folder scope for the file.", type="string"),
        AttributeInfo(name="page", description="Page number for page-based sources such as PDFs.", type="integer"),
    ]
    today = _today()
    yesterday = today - timedelta(days=1)
    examples = [
        (
            "give me log files from yesterday",
            {
                "query": "log files",
                "filter": f"and(eq('document_kind', 'log'), eq('source_mtime_date', '{yesterday.isoformat()}'))",
            },
        ),
        (
            "give me all the books I have by Franz Kafka",
            {
                "query": "books",
                "filter": "and(eq('document_kind', 'book'), eq('author', 'Franz Kafka'))",
            },
        ),
        (
            "search in error_report.log",
            {
                "query": "search",
                "filter": "eq('file_name', 'error_report.log')",
            },
        ),
    ]
    return load_query_constructor_runnable(
        llm=_get_filter_llm(),
        document_contents="A local multimodal index of personal files, books, logs, screenshots, and documents.",
        attribute_info=attribute_info,
        examples=examples,
    )


def _fallback_structured_filter(query: str):
    lowered = query.lower()
    filters = []
    normalized_query = _normalize_relative_dates(query)

    date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", normalized_query)
    if date_match:
        filters.append(
            Comparison(
                comparator=Comparator.EQ,
                attribute="source_mtime_date",
                value=date_match.group(1),
            )
        )

    if "log" in lowered:
        filters.append(
            Comparison(comparator=Comparator.EQ, attribute="document_kind", value="log")
        )

    if "book" in lowered or "books" in lowered:
        filters.append(
            Comparison(comparator=Comparator.EQ, attribute="document_kind", value="book")
        )

    by_match = re.search(r"\bby\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)", query)
    if by_match:
        filters.append(
            Comparison(comparator=Comparator.EQ, attribute="author", value=by_match.group(1).strip())
        )

    filename_match = _FILENAME_PATTERN.search(query)
    if filename_match:
        filters.append(
            Comparison(comparator=Comparator.EQ, attribute="file_name", value=filename_match.group(1))
        )

    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return Operation(operator=Operator.AND, arguments=filters)


def compile_multimodal_filter(query: str, file_filter: Optional[str] = None, workspace_id: Optional[str] = None) -> Tuple[str, Optional[str]]:
    normalized_query = _normalize_relative_dates(query)
    structured_query = None
    try:
        structured_query = _get_query_constructor().invoke({"query": normalized_query})
    except Exception as exc:
        print(f"⚠️ Self-query constructor failed, using fallback metadata parsing: {exc}")

    if structured_query is None:
        class _Structured:
            query = normalized_query
            filter = None
        structured_query = _Structured()

    if not getattr(structured_query, "filter", None):
        structured_query.filter = _fallback_structured_filter(normalized_query)

    extra_filters = []
    if file_filter:
        extra_filters.append(
            Comparison(
                comparator=Comparator.EQ,
                attribute="source_path",
                value=os.path.abspath(file_filter),
            )
        )
    if workspace_id:
        extra_filters.append(
            Comparison(
                comparator=Comparator.EQ,
                attribute="workspace_id",
                value=workspace_id,
            )
        )

    base_filter = getattr(structured_query, "filter", None)
    all_filters = [item for item in ([base_filter] + extra_filters) if item is not None]
    if not all_filters:
        return structured_query.query or normalized_query, None

    final_filter = all_filters[0] if len(all_filters) == 1 else Operation(operator=Operator.AND, arguments=all_filters)
    structured_query.filter = final_filter
    text_query, kwargs = LanceDBTranslator().visit_structured_query(structured_query)
    return text_query or normalized_query, kwargs.get("filter")
