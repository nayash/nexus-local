import os
import re
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.structured_query import Comparator, Comparison, Operation, Operator
from langchain_ollama import ChatOllama

from src.core.config import Config
from src.core.user_settings import get_setting
from src.rag.ingestion_multimodal import search_multimodal
from src.rag.lancedb_store import load_rows
from src.rag.query_filters import build_multimodal_structured_query
from src.rag.schemas import parse_extra
from src.tools.schemas import SearchResult
from src.tools.tool_results import build_final_response_artifact


_DIRECT_RESPONSE_CHAR_LIMIT = 100000
_LOCAL_RETRIEVAL_PLAN_PROMPT = """You are a retrieval planner for a local document search engine.

Classify the user's request into one of two retrieval modes:
- "document_lookup": The user is asking to identify, list, or select files/documents by metadata such as file name, title, author, type, date, or explicit references to the documents table.
- "semantic_search": The user is asking for information from inside document contents.

Classify the response mode:
- "full_document": The user wants the entire file/document text returned.
- "snippets": The user wants an answer or relevant excerpts rather than the entire document.

Return ONLY valid JSON with exactly this shape:
{"retrieval_mode":"document_lookup|semantic_search","response_mode":"full_document|snippets"}
"""
_PLANNER_CACHE = {}
_RELEVANCE_STOPWORDS = {
    "the",
    "a",
    "an",
    "for",
    "from",
    "my",
    "me",
    "local",
    "please",
    "give",
    "find",
    "extract",
    "show",
    "tell",
    "about",
}
_NOTE_INTENT_TERMS = {"note", "notes", "writing", "story", "tips", "idea", "ideas"}


def _normalize_whitespace(value: str) -> str:
    return " ".join((value or "").split())


def _query_terms(query: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]{1,}", (query or "").lower()):
        if len(token) < 3 or token in _RELEVANCE_STOPWORDS:
            continue
        terms.add(token)
    return terms


def _text_relevance_overlap_score(query_terms: set[str], text_blob: str) -> float:
    if not query_terms:
        return 0.0
    lowered_blob = (text_blob or "").lower()
    matched = sum(1 for term in query_terms if term in lowered_blob)
    return matched / max(len(query_terms), 1)


def _intent_source_bonus(row: dict, query_terms: set[str]) -> float:
    if not query_terms:
        return 0.0

    source_type = str(row.get("source_type") or "").lower()
    source_path = str(row.get("source_path") or "").lower()
    file_name = str(row.get("file_name") or "").lower()

    bonus = 0.0
    if any(term in _NOTE_INTENT_TERMS for term in query_terms):
        # Prefer note-like files for note/writing intents.
        if source_type in {"md", "txt", "docx"}:
            bonus += 0.25
        if source_type == "pdf":
            bonus -= 0.20

    if any(term in source_path for term in query_terms):
        bonus += 0.20
    if any(term in file_name for term in query_terms):
        bonus += 0.20

    return bonus


def _rerank_and_filter_rows(rows: list[dict], query: str, top_k: int) -> list[dict]:
    if not rows:
        return []

    query_terms = _query_terms(query)
    if not query_terms:
        return rows[:top_k]

    scored = []
    for index, row in enumerate(rows):
        extra = parse_extra(row.get("extra"))
        text_blob = " ".join(
            [
                str(row.get("title") or ""),
                str(row.get("file_name") or ""),
                str(row.get("source_path") or ""),
                str(extra.get("matched_text") or ""),
                str(row.get("text") or ""),
            ]
        )
        overlap = _text_relevance_overlap_score(query_terms, text_blob)
        intent_bonus = _intent_source_bonus(row, query_terms)
        base_score = float(row.get("_score") or 0.0)
        final_score = base_score + (1.2 * overlap) + intent_bonus
        scored.append((final_score, overlap, index, row))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    selected = []
    seen_paths = set()
    for final_score, overlap, _, row in scored:
        source_path = os.path.abspath(row.get("source_path") or "")
        if source_path in seen_paths:
            continue

        # Keep only rows with lexical support, unless nothing else is available.
        if overlap <= 0 and len(selected) >= 2:
            continue

        selected.append(row)
        if source_path:
            seen_paths.add(source_path)
        if len(selected) >= top_k:
            break

    return selected or [item[3] for item in scored[: max(1, min(top_k, 2))]]


def _trim_excerpt(text: str, limit: int) -> str:
    cleaned = _normalize_whitespace(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _excerpt_around_match(full_text: str, matched_text: str, window: int = 700) -> str:
    cleaned_full = _normalize_whitespace(full_text)
    cleaned_match = _normalize_whitespace(matched_text)
    if not cleaned_full:
        return ""
    if not cleaned_match:
        return _trim_excerpt(cleaned_full, window)

    full_lower = cleaned_full.lower()
    match_lower = cleaned_match.lower()
    index = full_lower.find(match_lower)
    if index == -1:
        return _trim_excerpt(cleaned_full, window)

    half_window = max(window // 2, len(cleaned_match))
    start = max(0, index - half_window)
    end = min(len(cleaned_full), index + len(cleaned_match) + half_window)
    excerpt = cleaned_full[start:end].strip()
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(cleaned_full):
        excerpt = excerpt + "..."
    return excerpt


def _build_text_result_content(row: dict) -> str:
    full_text = (row.get("text") or "").strip()
    extra = parse_extra(row.get("extra"))
    matched_text = (extra.get("matched_text") or "").strip()
    page = row.get("page")

    sections = []
    if page:
        sections.append(f"Page: {page}")
    if matched_text:
        sections.append(f"Most relevant excerpt: {_trim_excerpt(matched_text, 320)}")
    if full_text:
        sections.append(
            f"Supporting passage: {_excerpt_around_match(full_text, matched_text, window=900)}"
        )
    return "\n".join(section for section in sections if section)


def _get_retrieval_planner():
    model_name = get_setting("model_name", "llama3.1")
    planner = _PLANNER_CACHE.get(model_name)
    if planner is None:
        planner = ChatOllama(
            model=model_name,
            temperature=0,
            base_url=Config.OLLAMA_BASE_URL,
        )
        _PLANNER_CACHE[model_name] = planner
    return planner


def _extract_json_object(raw_text: str) -> dict:
    import json

    candidate = (raw_text or "").strip()
    if not candidate:
        return {}
    if "{" in candidate and "}" in candidate:
        start = candidate.find("{")
        end = candidate.rfind("}") + 1
        if end > start:
            candidate = candidate[start:end]
    try:
        parsed = json.loads(candidate)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_retrieval_plan(plan: dict) -> dict:
    retrieval_mode = str(plan.get("retrieval_mode", "")).strip().lower()
    response_mode = str(plan.get("response_mode", "")).strip().lower()

    if retrieval_mode not in {"document_lookup", "semantic_search"}:
        retrieval_mode = "semantic_search"
    if response_mode not in {"full_document", "snippets"}:
        response_mode = "snippets"

    return {
        "retrieval_mode": retrieval_mode,
        "response_mode": response_mode,
    }


def _plan_local_retrieval(query: str, file_filter: str = "") -> dict:
    context_line = (
        f"Focused file: {os.path.abspath(file_filter)}"
        if file_filter
        else "Focused file: none"
    )
    messages = [
        SystemMessage(content=_LOCAL_RETRIEVAL_PLAN_PROMPT),
        HumanMessage(content=f"{context_line}\nUser query: {query}"),
    ]

    try:
        response = _get_retrieval_planner().invoke(messages)
        plan = _normalize_retrieval_plan(_extract_json_object(getattr(response, "content", "") or ""))
    except Exception as exc:
        print(f"⚠️ Local retrieval planner failed: {exc}")
        plan = {"retrieval_mode": "semantic_search", "response_mode": "snippets"}

    print(f"local retrieval plan: {plan}")
    return plan


def _source_type_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".pdf": "pdf",
        ".docx": "docx",
        ".txt": "txt",
        ".log": "log",
        ".md": "md",
        ".csv": "csv",
        ".html": "html",
        ".htm": "html",
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
    }.get(ext, ext.lstrip(".") or "unknown")


def _source_title(source_path: str, source_type: Optional[str] = None) -> str:
    base_name = os.path.basename(source_path) or "Unknown"
    resolved_type = source_type or _source_type_from_path(source_path)
    return f"Local File ({resolved_type}): {base_name}"


def _build_source_metadata(rows: list[dict]) -> list[dict]:
    metadata = []
    for row in rows:
        source_path = os.path.abspath(row.get("source_path") or "")
        if not source_path:
            continue
        metadata.append(
            {
                "title": _source_title(source_path, row.get("source_type")),
                "url": source_path,
                "type": "local",
            }
        )
    return metadata


def _like_pattern_to_regex(pattern: str) -> str:
    escaped = re.escape(pattern or "")
    escaped = escaped.replace("%", ".*").replace("_", ".")
    return f"^{escaped}$"


def _row_matches_comparison(row: dict, comparison: Comparison) -> bool:
    row_value = row.get(comparison.attribute)
    expected_value = comparison.value
    comparator = comparison.comparator

    if comparator == Comparator.EQ:
        return row_value == expected_value
    if comparator == Comparator.NE:
        return row_value != expected_value
    if comparator == Comparator.LIKE:
        if row_value is None:
            return False
        return bool(
            re.match(
                _like_pattern_to_regex(str(expected_value)),
                str(row_value),
                flags=re.IGNORECASE,
            )
        )
    if comparator == Comparator.IN:
        if isinstance(expected_value, list):
            return row_value in expected_value
        return False

    if row_value is None:
        return False

    left = row_value
    right = expected_value
    if isinstance(left, str) and not isinstance(right, str):
        try:
            left = type(right)(left)
        except Exception:
            pass

    if comparator == Comparator.GT:
        return left > right
    if comparator == Comparator.GTE:
        return left >= right
    if comparator == Comparator.LT:
        return left < right
    if comparator == Comparator.LTE:
        return left <= right
    return False


def _row_matches_filter(row: dict, filter_node) -> bool:
    if filter_node is None:
        return True
    if isinstance(filter_node, Comparison):
        return _row_matches_comparison(row, filter_node)
    if isinstance(filter_node, Operation):
        if filter_node.operator == Operator.AND:
            return all(_row_matches_filter(row, arg) for arg in filter_node.arguments)
        if filter_node.operator == Operator.OR:
            return any(_row_matches_filter(row, arg) for arg in filter_node.arguments)
        if filter_node.operator == Operator.NOT:
            if not filter_node.arguments:
                return True
            return not _row_matches_filter(row, filter_node.arguments[0])
    return False


def _limit_direct_payload(text: str, limit: int = _DIRECT_RESPONSE_CHAR_LIMIT) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return (
        text[:limit].rstrip()
        + f"\n\n... [Output truncated after {limit} characters for display] ...",
        True,
    )


def _append_with_overlap(existing: str, chunk: str, max_overlap: int = 400) -> str:
    if not existing:
        return chunk
    if not chunk:
        return existing

    overlap_limit = min(len(existing), len(chunk), max_overlap)
    for size in range(overlap_limit, 0, -1):
        if existing[-size:] == chunk[:size]:
            return existing + chunk[size:]
    return existing + "\n" + chunk


def _read_text_source(path: str, source_type: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            content = handle.read()

    if source_type == "html":
        try:
            from bs4 import BeautifulSoup

            content = BeautifulSoup(content, "html.parser").get_text("\n")
        except Exception:
            content = re.sub(r"<[^>]+>", " ", content)
    return content


def _read_pdf_source(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    parts = []
    for page_index, page in enumerate(reader.pages, start=1):
        extracted = (page.extract_text() or "").strip()
        if not extracted:
            continue
        parts.append(f"--- Page {page_index} ---\n{extracted}")
    return "\n\n".join(parts).strip()


def _read_docx_source(path: str) -> str:
    from docx import Document as DocxDocument

    document = DocxDocument(path)
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
    return "\n".join(paragraphs).strip()


def _reconstruct_text_from_index(source_path: str) -> str:
    target_path = os.path.abspath(source_path)
    rows = [
        row
        for row in load_rows(Config.MULTIMODAL_PARENT_TABLE)
        if os.path.abspath(row.get("source_path") or "") == target_path
        and row.get("modality", "text") == "text"
        and (row.get("text") or "").strip()
    ]
    if not rows:
        return ""

    def _sort_key(row: dict):
        page = row.get("page")
        page_value = page if isinstance(page, int) and page is not None else -1
        parent_index = row.get("parent_index")
        parent_value = parent_index if isinstance(parent_index, int) and parent_index is not None else -1
        return page_value, parent_value

    grouped: dict[int, list[str]] = {}
    for row in sorted(rows, key=_sort_key):
        page = row.get("page")
        page_key = page if isinstance(page, int) and page is not None else -1
        grouped.setdefault(page_key, []).append((row.get("text") or "").strip())

    page_texts = []
    for page_key in sorted(grouped):
        merged = ""
        for chunk in grouped[page_key]:
            merged = _append_with_overlap(merged, chunk)
        if not merged:
            continue
        if page_key >= 0:
            page_texts.append(f"--- Page {page_key} ---\n{merged}")
        else:
            page_texts.append(merged)
    return "\n\n".join(page_texts).strip()


def _load_full_text(source_path: str, source_type: str) -> str:
    abs_path = os.path.abspath(source_path)
    if os.path.exists(abs_path):
        try:
            if source_type in {"txt", "log", "md", "csv", "html"}:
                return _read_text_source(abs_path, source_type)
            if source_type == "pdf":
                return _read_pdf_source(abs_path)
            if source_type == "docx":
                return _read_docx_source(abs_path)
        except Exception as exc:
            print(f"⚠️ Failed to read source file '{abs_path}': {exc}")

    return _reconstruct_text_from_index(abs_path)


def _query_documents_table(query: str, file_filter: str = "") -> list[dict]:
    normalized_file_filter = (file_filter or "").strip() or None
    structured_query, _, _ = build_multimodal_structured_query(
        query,
        file_filter=normalized_file_filter,
    )
    filter_node = getattr(structured_query, "filter", None)
    if filter_node is None:
        return []

    rows = []
    for row in load_rows(Config.MULTIMODAL_DOCUMENTS_TABLE):
        source_path = os.path.abspath(row.get("source_path") or "")
        if not source_path:
            continue
        normalized_row = dict(row)
        normalized_row["source_path"] = source_path
        if _row_matches_filter(normalized_row, filter_node):
            rows.append(normalized_row)

    deduped = {}
    for row in rows:
        deduped[row["source_path"]] = row
    return sorted(
        deduped.values(),
        key=lambda item: (
            (item.get("file_name") or "").lower(),
            (item.get("source_path") or "").lower(),
        ),
    )


def resolve_direct_local_response(
    query: str, file_filter: str = ""
) -> Optional[tuple[str, list[dict]]]:
    normalized_file_filter = (file_filter or "").strip()
    plan = _plan_local_retrieval(query, normalized_file_filter)
    if plan["retrieval_mode"] != "document_lookup":
        return None

    matches = _query_documents_table(query, normalized_file_filter)
    wants_full_content = plan["response_mode"] == "full_document"

    metadata = _build_source_metadata(matches)

    if not matches:
        return (
            "",
            metadata + [build_final_response_artifact("No indexed files matched the requested document lookup.")],
        )

    if len(matches) > 1:
        lines = [f"Found {len(matches)} matching files:"]
        for row in matches[:20]:
            lines.append(row["source_path"])
        if len(matches) > 20:
            lines.append(f"... and {len(matches) - 20} more")
        lines.append("")
        lines.append("Specify the exact filename or focus the file to continue.")
        return "", metadata + [build_final_response_artifact("\n".join(lines))]

    match = matches[0]
    source_path = match["source_path"]

    if not wants_full_content:
        return "", metadata + [build_final_response_artifact(f"Matched file:\n{source_path}")]

    source_type = (match.get("source_type") or _source_type_from_path(source_path)).lower()
    if source_type == "image":
        return (
            "",
            metadata
            + [
                build_final_response_artifact(
                    f"Matched file:\n{source_path}\n\nThis file is an image and does not have extractable text content."
                )
            ],
        )

    full_text = _load_full_text(source_path, source_type)
    if not full_text.strip():
        return (
            "",
            metadata
            + [
                build_final_response_artifact(
                    f"Matched file:\n{source_path}\n\nThe file was found, but its text content could not be reconstructed."
                )
            ],
        )

    display_text, truncated = _limit_direct_payload(full_text)
    prefix = f"Full content of {source_path}:\n\n"
    if truncated:
        prefix += "[The file exceeds the direct display limit, so the output below is truncated.]\n\n"
    return "", metadata + [build_final_response_artifact(prefix + display_text)]


def search_local(query: str, file_filter: str = "") -> List[SearchResult]:
    """
    Searches the local LanceDB for relevant document chunks using the active strategy.
    Args:
        query: The search text.
        file_filter: Optional absolute path to restrict search to a specific file.
    """
    
    file_filter = (file_filter or "").strip() or None

    # Validation: Ensure focused file actually exists
    if file_filter and not os.path.exists(file_filter):
        print(f"⚠️ WARNING: Intent to focus on '{file_filter}' ignored because file does not exist.")
        file_filter = None

    if not Config.MULTIMODAL_EMBEDDINGS_ENABLED:
        return [SearchResult(title="System", url="local", content="No local knowledge found. Please ingest files or folders first.")]

    try:
        top_k = 10 if not file_filter else 20
        multimodal_results = _search_multimodal_results(query, file_filter=file_filter, top_k=top_k)
    except Exception as exc:
        print(f"⚠️ Multimodal search failed: {exc}")
        multimodal_results = []

    if not multimodal_results:
        msg = "No relevant local documents found."
        if file_filter:
            msg = f"The focused document '{os.path.basename(file_filter)}' does not seem to contain information regarding your query."
        return [SearchResult(title="Info", url="", content=msg)]

    return multimodal_results


def _search_multimodal_results(query: str, file_filter: str = None, top_k: int = 10) -> List[SearchResult]:
    rows = search_multimodal(query, top_k=top_k, file_filter=file_filter)
    if not rows:
        return []
    # Keep context focused by reranking and deduplicating per source file.
    rows = _rerank_and_filter_rows(rows, query=query, top_k=min(top_k, 5))

    results = []
    for row in rows:
        source_path = row.get("source_path") or ""
        if file_filter and os.path.abspath(source_path) != os.path.abspath(file_filter):
            continue

        modality = row.get("modality", "text")
        source_type = row.get("source_type", "local")
        base_name = os.path.basename(source_path) or "Unknown"

        if modality == "image":
            extra = row.get("extra") or "{}"
            placeholder = (
                f"[IMAGE] file={source_path} "
                f"page={row.get('page')} image_index={row.get('image_index')} "
                f"mime={row.get('mime')} size={row.get('width')}x{row.get('height')} "
                f"cached={extra}"
            )
            content = placeholder
            title = f"Local Image ({source_type}): {base_name}"
        else:
            content = _build_text_result_content(row)
            title = f"Local File ({source_type}): {base_name}"

        results.append(
            SearchResult(
                title=title,
                url=source_path,
                content=content,
                source="local",
            )
        )

    return results
