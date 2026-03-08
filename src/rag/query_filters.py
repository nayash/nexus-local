import os
import re
import json
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from typing import Optional, Tuple

from langchain_classic.chains.query_constructor.base import AttributeInfo, load_query_constructor_runnable
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
)
from langchain_ollama import ChatOllama

from src.core.config import Config
from src.core.user_settings import get_setting
from src.rag.metadata_taxonomy import (
    CANONICAL_DOCUMENT_KINDS,
    infer_document_kind_from_query,
    is_broad_personal_corpus_query,
    normalize_document_kind,
)
from src.rag.translators import LanceDBTranslator


_FILENAME_PATTERN = re.compile(
    r"\b([\w\-. ]+\.(pdf|txt|md|csv|sh|py|docx|html|htm|jpg|jpeg|png|log))\b",
    re.IGNORECASE,
)
_ALLOWED_FILTER_ATTRIBUTES = {
    "file_name",
    "source_path",
    "source_type",
    "document_kind",
    "author",
    "title",
    "owner",
    "source_mtime_date",
    "workspace_id",
    "page",
}
_ALLOWED_COMPARATORS = {
    Comparator.EQ,
    Comparator.NE,
    Comparator.GT,
    Comparator.GTE,
    Comparator.LT,
    Comparator.LTE,
    Comparator.IN,
    Comparator.LIKE,
}
_QUOTED_TEXT_PATTERN = re.compile(r"\"([^\"]{3,140})\"|'([^']{3,140})'")
_CORRECTIVE_CUE_PATTERN = re.compile(
    r"\b(i asked you to|i asked for|my question was|i wanted|you were supposed to)\b",
    re.IGNORECASE,
)
_QUERY_NORMALIZER_PROMPT = """You normalize local document retrieval queries.

Return ONLY valid JSON with this exact schema:
{"clean_query":"string","focus_phrases":["string"],"confidence":"high|medium|low"}

Rules:
- Remove complaint/meta conversation about assistant behavior.
- Keep only the actionable retrieval request.
- Preserve filenames, titles, entities, and quoted targets.
- If there is an explicit target phrase/title, include it in focus_phrases.
- Do not invent metadata filters or extra fields.
"""


@dataclass
class FilterClauseDecision:
    attribute: str
    comparator: str
    value: str
    confidence: str
    reason: str
    normalized_from: Optional[str] = None
    dropped: bool = False


@dataclass
class CompiledFilterPlan:
    text_query: str
    strict_sql_filter: Optional[str]
    relaxed_sql_filter: Optional[str]
    should_try_relaxed: bool
    strict_clauses: list[FilterClauseDecision]
    dropped_clauses: list[FilterClauseDecision]
    allow_unfiltered_fallback: bool = True


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
def _get_query_normalizer_llm():
    return ChatOllama(
        model=get_setting("model_name", "llama3.1"),
        temperature=0,
        base_url=Config.OLLAMA_BASE_URL,
    )


@lru_cache(maxsize=1)
def _get_query_constructor():
    allowed_kinds = ", ".join(CANONICAL_DOCUMENT_KINDS)
    attribute_info = [
        AttributeInfo(name="file_name", description="The source filename including extension.", type="string"),
        AttributeInfo(name="source_path", description="Absolute source path for the ingested file.", type="string"),
        AttributeInfo(name="source_type", description="The file type such as pdf, docx, csv, txt, image, html.", type="string"),
        AttributeInfo(
            name="document_kind",
            description=(
                f"High-level content kind. Allowed values: {allowed_kinds}. "
                "Treat note/notes/notebook/memo/journal as document."
            ),
            type="string",
        ),
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
            "give me writing ideas from my notes",
            {
                "query": "writing ideas",
                "filter": "eq('document_kind', 'document')",
            },
        ),
        (
            "search in error_report.log",
            {
                "query": "search",
                "filter": "eq('file_name', 'error_report.log')",
            },
        ),
        (
            'look for the file with name containing "nexus" and "logs" in the documents table',
            {
                "query": "file",
                "filter": "and(like('file_name', '%nexus%'), like('file_name', '%logs%'))",
            },
        ),
        (
            "extract full content of nexus-local-logs-debug.txt",
            {
                "query": "full content",
                "filter": "eq('file_name', 'nexus-local-logs-debug.txt')",
            },
        ),
    ]
    return load_query_constructor_runnable(
        llm=_get_filter_llm(),
        document_contents="A local multimodal index of personal files, books, logs, screenshots, and documents.",
        attribute_info=attribute_info,
        examples=examples,
    )


def _extract_json_object(raw_text: str) -> dict:
    candidate = (raw_text or "").strip()
    if not candidate:
        return {}

    if "{" in candidate and "}" in candidate:
        start = candidate.find("{")
        end = candidate.rfind("}") + 1
        if end > start:
            candidate = candidate[start:end]

    try:
        payload = json.loads(candidate)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_focus_phrase(value: str) -> str:
    candidate = " ".join((value or "").split()).strip().strip(".,:;!?")
    if len(candidate) < 3 or len(candidate) > 120:
        return ""
    if not re.search(r"[A-Za-z0-9]", candidate):
        return ""
    return candidate


def _extract_quoted_focus_phrases(query: str) -> list[str]:
    phrases = []
    seen = set()
    for match in _QUOTED_TEXT_PATTERN.finditer(query or ""):
        raw_phrase = (match.group(1) or match.group(2) or "").strip()
        phrase = _normalize_focus_phrase(raw_phrase)
        lowered = phrase.lower()
        if phrase and lowered not in seen:
            phrases.append(phrase)
            seen.add(lowered)
    return phrases


def _coerce_focus_phrases(raw_value) -> list[str]:
    candidates = []
    if isinstance(raw_value, str):
        candidates = [raw_value]
    elif isinstance(raw_value, list):
        candidates = [item for item in raw_value if isinstance(item, str)]

    phrases = []
    seen = set()
    for item in candidates:
        phrase = _normalize_focus_phrase(item)
        lowered = phrase.lower()
        if phrase and lowered not in seen:
            phrases.append(phrase)
            seen.add(lowered)
    return phrases


def _merge_focus_phrases(*phrase_groups: list[str]) -> list[str]:
    merged = []
    seen = set()
    for group in phrase_groups:
        for phrase in group or []:
            normalized = _normalize_focus_phrase(phrase)
            lowered = normalized.lower()
            if normalized and lowered not in seen:
                merged.append(normalized)
                seen.add(lowered)
    return merged


def _deterministic_corrective_rewrite(query: str) -> str:
    normalized = " ".join((query or "").split()).strip()
    if not normalized:
        return ""

    match = _CORRECTIVE_CUE_PATTERN.search(normalized)
    if not match:
        return normalized

    tail = normalized[match.end():].strip(" :-,.;")
    if len(tail.split()) < 2:
        return normalized
    return tail


def _llm_normalize_query(query: str) -> tuple[str, list[str], str, bool]:
    messages = [
        SystemMessage(content=_QUERY_NORMALIZER_PROMPT),
        HumanMessage(content=query),
    ]
    try:
        response = _get_query_normalizer_llm().invoke(messages)
    except Exception as exc:
        print(f"⚠️ Query normalizer failed: {exc}")
        fallback_query = _deterministic_corrective_rewrite(query) or query
        fallback_phrases = _extract_quoted_focus_phrases(query)
        if fallback_query != query:
            print(f"query normalizer deterministic rewrite | clean_query={fallback_query!r}")
        return fallback_query, fallback_phrases, "low", False

    payload = _extract_json_object(getattr(response, "content", "") or "")
    if not payload:
        print("⚠️ Query normalizer returned invalid JSON; using deterministic fallback.")
        fallback_query = _deterministic_corrective_rewrite(query) or query
        fallback_phrases = _extract_quoted_focus_phrases(query)
        if fallback_query != query:
            print(f"query normalizer deterministic rewrite | clean_query={fallback_query!r}")
        return fallback_query, fallback_phrases, "low", False

    clean_query = " ".join(str(payload.get("clean_query", "") or "").split()).strip() or query
    focus_phrases = _coerce_focus_phrases(payload.get("focus_phrases"))
    if not focus_phrases and isinstance(payload.get("focus_phrase"), str):
        focus_phrases = _coerce_focus_phrases(payload.get("focus_phrase"))

    confidence = str(payload.get("confidence", "low") or "low").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    return clean_query, focus_phrases, confidence, True


def _build_focus_phrase_filter(phrases: list[str]):
    if not phrases:
        return None

    phrase_filters = []
    for phrase in phrases:
        like_value = f"%{phrase}%"
        comparisons = [
            Comparison(comparator=Comparator.LIKE, attribute="title", value=like_value),
            Comparison(comparator=Comparator.LIKE, attribute="file_name", value=like_value),
            Comparison(comparator=Comparator.LIKE, attribute="source_path", value=like_value),
        ]
        if re.search(r"\.[A-Za-z0-9]{1,8}$", phrase):
            comparisons.insert(
                0,
                Comparison(comparator=Comparator.EQ, attribute="file_name", value=phrase),
            )
        phrase_filters.append(
            comparisons[0]
            if len(comparisons) == 1
            else Operation(operator=Operator.OR, arguments=comparisons)
        )

    if len(phrase_filters) == 1:
        return phrase_filters[0]
    return Operation(operator=Operator.AND, arguments=phrase_filters)


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

    inferred_kind = infer_document_kind_from_query(lowered)
    if inferred_kind in {"log", "book", "data", "image", "code", "document"}:
        filters.append(
            Comparison(comparator=Comparator.EQ, attribute="document_kind", value=inferred_kind)
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


class _FallbackStructuredQuery:
    def __init__(self, query: str):
        self.query = query
        self.filter = None


def _stringify_filter_value(value) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _has_explicit_date(query: str) -> bool:
    lowered = query.lower()
    if re.search(r"\b20\d{2}-\d{2}-\d{2}\b", query):
        return True
    return "today" in lowered or "yesterday" in lowered


def _comparison_confidence(comparison: Comparison, original_query: str) -> tuple[str, str]:
    attribute = comparison.attribute
    value_str = _stringify_filter_value(comparison.value).lower()
    query_lower = original_query.lower()

    if attribute in {"source_path", "workspace_id", "file_name", "title"}:
        return "high", "Explicit file/workspace/title constraint"

    if attribute == "source_mtime_date":
        if _has_explicit_date(original_query):
            return "high", "Query includes explicit date intent"
        return "low", "Date filter inferred without explicit date intent"

    if attribute == "author":
        if " by " in f" {query_lower} ":
            return "high", "Author intent is explicit"
        return "low", "Author filter inferred without explicit intent"

    if attribute == "source_type":
        if value_str in query_lower:
            return "high", "Source type explicitly mentioned"
        return "low", "Source type inferred from semantics"

    if attribute == "document_kind":
        inferred_kind = infer_document_kind_from_query(original_query)
        if inferred_kind and inferred_kind == value_str:
            confidence = "high"
            reason = "Document kind explicitly requested"
        else:
            confidence = "low"
            reason = "Document kind inferred by planner"

        if is_broad_personal_corpus_query(original_query):
            confidence = "low"
            reason = "Broad personal corpus query; avoid aggressive metadata filtering"
        return confidence, reason

    if attribute == "page":
        if "page" in query_lower:
            return "high", "Page constraint explicitly requested"
        return "low", "Page filter inferred by planner"

    return "low", "Planner-inferred metadata filter"


def _normalize_comparison(
    comparison: Comparison,
    original_query: str,
    decisions: list[FilterClauseDecision],
) -> Optional[Comparison]:
    attribute = (comparison.attribute or "").strip()
    comparator = comparison.comparator
    value = comparison.value

    if attribute not in _ALLOWED_FILTER_ATTRIBUTES:
        decisions.append(
            FilterClauseDecision(
                attribute=attribute or "<missing>",
                comparator=comparator.value,
                value=_stringify_filter_value(value),
                confidence="low",
                reason="Unsupported filter attribute; dropped",
                dropped=True,
            )
        )
        return None

    if comparator not in _ALLOWED_COMPARATORS:
        decisions.append(
            FilterClauseDecision(
                attribute=attribute,
                comparator=comparator.value,
                value=_stringify_filter_value(value),
                confidence="low",
                reason="Unsupported comparator; dropped",
                dropped=True,
            )
        )
        return None

    normalized_from = None
    if attribute == "document_kind":
        normalized_from = _stringify_filter_value(value)
        normalized_kind = normalize_document_kind(normalized_from)
        if not normalized_kind:
            decisions.append(
                FilterClauseDecision(
                    attribute=attribute,
                    comparator=comparator.value,
                    value=normalized_from,
                    confidence="low",
                    reason="Invalid non-canonical document_kind; dropped",
                    dropped=True,
                )
            )
            return None
        value = normalized_kind

    confidence, reason = _comparison_confidence(
        Comparison(comparator=comparator, attribute=attribute, value=value),
        original_query,
    )
    decisions.append(
        FilterClauseDecision(
            attribute=attribute,
            comparator=comparator.value,
            value=_stringify_filter_value(value),
            confidence=confidence,
            reason=reason,
            normalized_from=normalized_from if normalized_from and normalized_from.lower() != str(value).lower() else None,
            dropped=False,
        )
    )
    return Comparison(comparator=comparator, attribute=attribute, value=value)


def _normalize_filter_tree(
    filter_node,
    original_query: str,
    decisions: list[FilterClauseDecision],
):
    if filter_node is None:
        return None

    if isinstance(filter_node, Comparison):
        return _normalize_comparison(filter_node, original_query, decisions)

    if isinstance(filter_node, Operation):
        normalized_args = []
        for argument in filter_node.arguments:
            normalized = _normalize_filter_tree(argument, original_query, decisions)
            if normalized is not None:
                normalized_args.append(normalized)

        if not normalized_args:
            return None

        if filter_node.operator == Operator.NOT:
            return Operation(operator=Operator.NOT, arguments=[normalized_args[0]])

        if len(normalized_args) == 1:
            return normalized_args[0]

        return Operation(operator=filter_node.operator, arguments=normalized_args)

    return None


def _decision_signature(decision: FilterClauseDecision) -> tuple[str, str, str]:
    return decision.attribute, decision.comparator, decision.value


def _comparison_signature(comparison: Comparison) -> tuple[str, str, str]:
    return (
        comparison.attribute,
        comparison.comparator.value,
        _stringify_filter_value(comparison.value),
    )


def _relax_filter_tree(
    filter_node,
    low_confidence_signatures: set[tuple[str, str, str]],
):
    if filter_node is None:
        return None

    if isinstance(filter_node, Comparison):
        signature = _comparison_signature(filter_node)
        if signature in low_confidence_signatures:
            return None
        return filter_node

    if isinstance(filter_node, Operation):
        relaxed_args = []
        for argument in filter_node.arguments:
            relaxed_arg = _relax_filter_tree(argument, low_confidence_signatures)
            if relaxed_arg is not None:
                relaxed_args.append(relaxed_arg)

        if not relaxed_args:
            return None

        if filter_node.operator == Operator.NOT:
            return Operation(operator=Operator.NOT, arguments=[relaxed_args[0]])

        if len(relaxed_args) == 1:
            return relaxed_args[0]

        return Operation(operator=filter_node.operator, arguments=relaxed_args)

    return None


def _compile_sql_filter(structured_query) -> Optional[str]:
    if not getattr(structured_query, "filter", None):
        return None
    _, kwargs = LanceDBTranslator().visit_structured_query(structured_query)
    return kwargs.get("filter")


def _build_structured_query_with_decisions(
    query: str,
    file_filter: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> tuple[object, str, Optional[str], list[FilterClauseDecision]]:
    normalized_query = _normalize_relative_dates(query)
    preprocessed_query = _deterministic_corrective_rewrite(normalized_query) or normalized_query
    if preprocessed_query != normalized_query:
        print(
            "query deterministic rewrite | "
            f"original={normalized_query!r} | rewritten={preprocessed_query!r}"
        )
    structured_query = None
    decisions: list[FilterClauseDecision] = []
    quoted_focus_phrases = _extract_quoted_focus_phrases(normalized_query)
    llm_normalized_query = preprocessed_query
    llm_focus_phrases: list[str] = []
    llm_confidence = "low"

    try:
        structured_query = _get_query_constructor().invoke({"query": preprocessed_query})
    except Exception as exc:
        print(f"⚠️ Self-query constructor failed, using fallback metadata parsing: {exc}")
        llm_normalized_query, llm_focus_phrases, llm_confidence, used_llm_normalizer = _llm_normalize_query(preprocessed_query)
        if used_llm_normalizer:
            print(
                "query normalizer resolved | "
                f"clean_query={llm_normalized_query!r} | "
                f"focus_phrases={llm_focus_phrases!r} | confidence={llm_confidence}"
            )
        else:
            llm_normalized_query = preprocessed_query

    if structured_query is None:
        structured_query = _FallbackStructuredQuery(llm_normalized_query)

    if not getattr(structured_query, "filter", None):
        structured_query.filter = _fallback_structured_filter(llm_normalized_query)

    focus_phrases = _merge_focus_phrases(quoted_focus_phrases, llm_focus_phrases)
    focus_phrase_filter = _build_focus_phrase_filter(focus_phrases)
    if focus_phrases:
        print(f"query focus phrases: {focus_phrases}")

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
    if focus_phrase_filter is not None:
        extra_filters.append(focus_phrase_filter)

    base_filter = getattr(structured_query, "filter", None)
    all_filters = [item for item in ([base_filter] + extra_filters) if item is not None]

    final_filter = all_filters[0] if len(all_filters) == 1 else Operation(operator=Operator.AND, arguments=all_filters) if all_filters else None
    normalized_filter = _normalize_filter_tree(final_filter, query, decisions)
    structured_query.filter = normalized_filter

    text_query = (str(structured_query.query or "") or llm_normalized_query or preprocessed_query).strip() or normalized_query
    sql_filter = _compile_sql_filter(structured_query)

    clause_summary = [
        f"{item.attribute} {item.comparator} {item.value} ({item.confidence})"
        for item in decisions
        if not item.dropped
    ]
    dropped_summary = [
        f"{item.attribute} {item.comparator} {item.value}"
        for item in decisions
        if item.dropped
    ]
    if clause_summary:
        print(f"self-query clauses: {clause_summary}")
    if dropped_summary:
        print(f"self-query dropped clauses: {dropped_summary}")

    print(f"self-query resolved | text_query={text_query!r} | sql_filter={sql_filter!r}")
    return structured_query, text_query, sql_filter, decisions


def build_multimodal_structured_query(
    query: str,
    file_filter: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> tuple[object, str, Optional[str]]:
    structured_query, text_query, sql_filter, _ = _build_structured_query_with_decisions(
        query,
        file_filter=file_filter,
        workspace_id=workspace_id,
    )
    return structured_query, text_query, sql_filter


def compile_multimodal_filter_plan(
    query: str,
    file_filter: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> CompiledFilterPlan:
    structured_query, text_query, strict_sql_filter, decisions = _build_structured_query_with_decisions(
        query,
        file_filter=file_filter,
        workspace_id=workspace_id,
    )

    strict_filter = getattr(structured_query, "filter", None)
    strict_clauses = [item for item in decisions if not item.dropped]
    precision_attributes = {"source_path", "workspace_id", "file_name", "title"}
    has_high_precision_filter = any(item.attribute in precision_attributes for item in strict_clauses)
    allow_unfiltered_fallback = not has_high_precision_filter
    if strict_filter is None:
        return CompiledFilterPlan(
            text_query=text_query,
            strict_sql_filter=None,
            relaxed_sql_filter=None,
            should_try_relaxed=False,
            strict_clauses=strict_clauses,
            dropped_clauses=[],
            allow_unfiltered_fallback=True,
        )

    low_confidence_signatures = {
        _decision_signature(item)
        for item in strict_clauses
        if item.confidence == "low"
    }
    if not low_confidence_signatures:
        return CompiledFilterPlan(
            text_query=text_query,
            strict_sql_filter=strict_sql_filter,
            relaxed_sql_filter=None,
            should_try_relaxed=False,
            strict_clauses=strict_clauses,
            dropped_clauses=[],
            allow_unfiltered_fallback=allow_unfiltered_fallback,
        )

    relaxed_filter = _relax_filter_tree(strict_filter, low_confidence_signatures)
    dropped_clauses = [item for item in strict_clauses if _decision_signature(item) in low_confidence_signatures]

    relaxed_sql_filter = None
    if relaxed_filter is not None:
        relaxed_query = _FallbackStructuredQuery(text_query)
        relaxed_query.filter = relaxed_filter
        relaxed_sql_filter = _compile_sql_filter(relaxed_query)

    should_try_relaxed = bool(dropped_clauses) and relaxed_sql_filter != strict_sql_filter
    print(
        "self-query filter-plan | "
        f"strict={strict_sql_filter!r} | relaxed={relaxed_sql_filter!r} | "
        f"dropped_low_conf={len(dropped_clauses)} | allow_unfiltered={allow_unfiltered_fallback}"
    )

    return CompiledFilterPlan(
        text_query=text_query,
        strict_sql_filter=strict_sql_filter,
        relaxed_sql_filter=relaxed_sql_filter,
        should_try_relaxed=should_try_relaxed,
        strict_clauses=strict_clauses,
        dropped_clauses=dropped_clauses,
        allow_unfiltered_fallback=allow_unfiltered_fallback,
    )


def compile_multimodal_filter(
    query: str,
    file_filter: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    plan = compile_multimodal_filter_plan(
        query,
        file_filter=file_filter,
        workspace_id=workspace_id,
    )
    return plan.text_query, plan.strict_sql_filter
