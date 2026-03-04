from src.core.config import Config
from src.tools.schemas import SearchResult
from src.rag.ingestion_multimodal import search_multimodal
from src.rag.schemas import parse_extra
from typing import List
import os


def _normalize_whitespace(value: str) -> str:
    return " ".join((value or "").split())


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
