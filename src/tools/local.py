import json
import os
import re
import time
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
from src.tools.tool_results import build_final_response_artifact, extract_final_response


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
_IDENTITY_FILE_NAME = "nexus-identity.txt"
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
_EXISTENCE_QUERY_HINTS = (
    "do i have",
    "did i have",
    "is there",
    "are there",
    "which",
    "list",
    "any",
)
_INVENTORY_QUERY_PATTERNS = (
    re.compile(r"\b(which|what|list|show)\s+(files|file names|filenames|documents|docs|notes|notebooks)\b"),
    re.compile(r"\b(do i have|did i have|are there|is there)\s+any\s+(files|documents|docs|notes|notebooks)\b"),
    re.compile(r"\bmatching\s+(files|documents|docs|notes|notebooks)\b"),
)
_LOCATE_QUERY_PATTERNS = (
    re.compile(r"\b(which|what)\s+file\b"),
    re.compile(r"\bwhere\s+(is|are)\b"),
    re.compile(r"\bfind\s+the\s+file\b"),
    re.compile(r"\bmatched\s+file\b"),
)
_FULL_DOCUMENT_QUERY_HINTS = (
    "full content",
    "full text",
    "entire file",
    "entire document",
    "whole file",
    "whole document",
    "complete file",
    "complete document",
)
_CONTENT_QUERY_HINTS = (
    "summarize",
    "summary",
    "explain",
    "analyze",
    "describe",
    "extract",
    "quote",
    "tell me",
    "give me",
    "show me",
    "list down",
    "key points",
    "writing ideas",
    "ideas i had",
    "what did i",
    "from my notes",
    "from my files",
    "from my documents",
    "in my notes",
    "in my files",
    "in my documents",
)
_SELF_IDENTITY_QUERY_HINTS = (
    "who are you",
    "what are you",
    "tell me about yourself",
    "introduce yourself",
    "what can you do",
    "what do you do",
    "what are your capabilities",
    "what are your limitations",
    "what cant you do",
    "what can t you do",
    "what can you not do",
    "who built you",
    "who made you",
    "who created you",
    "your mission",
    "are you offline",
    "are you private",
    "do you send my personal data",
    "do you send personal data",
    "what is your version",
    "what version are you",
    "what is your personality",
)

_LOCAL_RETRIEVAL_V2_PROMPT = """You are a local retrieval planner.

Classify the user request into exactly one mode:
- semantic_answer: answer from document contents/snippets.
- document_lookup: identify/list/locate files and metadata (path, title, author, type).
- catalog: list or count available files/documents from the indexed registry.
- full_document: return complete document text.
- hybrid: provide both semantic answer and compact matching-file inventory.

Return ONLY valid JSON:
{"mode":"semantic_answer|document_lookup|catalog|full_document|hybrid","reason":"short reason"}
"""


def _normalize_whitespace(value: str) -> str:
    return " ".join((value or "").split())


def _normalize_for_match(value: str) -> str:
    lowered = (value or "").lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _query_terms(query: str) -> set[str]:
    terms = set()
    normalized_query = _normalize_for_match(query)
    for token in normalized_query.split():
        if len(token) < 3 or token in _RELEVANCE_STOPWORDS:
            continue
        terms.add(token)
        # Small plural normalization improves lexical recall (idea <-> ideas).
        if token.endswith("s") and len(token) > 4:
            terms.add(token[:-1])
    return terms


def _text_relevance_overlap_score(query_terms: set[str], text_blob: str) -> float:
    if not query_terms:
        return 0.0
    lowered_blob = _normalize_for_match(text_blob or "")
    matched = sum(1 for term in query_terms if term in lowered_blob)
    return matched / max(len(query_terms), 1)


def _intent_source_bonus(row: dict, query_terms: set[str]) -> float:
    if not query_terms:
        return 0.0

    source_type = str(row.get("source_type") or "").lower()
    source_path = _normalize_for_match(str(row.get("source_path") or ""))
    file_name = _normalize_for_match(str(row.get("file_name") or ""))

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


def _is_existence_or_list_query(query: str) -> bool:
    normalized = _normalize_for_match(query)
    has_hint = any(hint in normalized for hint in _EXISTENCE_QUERY_HINTS)
    has_target = any(token in normalized for token in ("idea", "ideas", "game", "book", "file", "files", "note", "notes"))
    return has_hint and has_target


def _is_explicit_document_inventory_query(query: str) -> bool:
    normalized = _normalize_for_match(query)
    return any(pattern.search(normalized) for pattern in _INVENTORY_QUERY_PATTERNS)


def _is_explicit_file_location_query(query: str) -> bool:
    normalized = _normalize_for_match(query)
    return any(pattern.search(normalized) for pattern in _LOCATE_QUERY_PATTERNS)


def _is_content_extraction_query(query: str) -> bool:
    normalized = _normalize_for_match(query)
    if not normalized:
        return False
    if _is_explicit_document_inventory_query(normalized) or _is_explicit_file_location_query(normalized):
        return False
    return any(hint in normalized for hint in _CONTENT_QUERY_HINTS)


def _wants_full_document(query: str) -> bool:
    normalized = _normalize_for_match(query)
    return any(hint in normalized for hint in _FULL_DOCUMENT_QUERY_HINTS)


def _document_metadata_answer(query: str, match: dict) -> Optional[str]:
    normalized = _normalize_for_match(query)
    source_path = os.path.abspath(match.get("source_path") or "")
    file_name = match.get("file_name") or os.path.basename(source_path)
    source_type = (match.get("source_type") or _source_type_from_path(source_path)).lower()

    if "author" in normalized or "who wrote" in normalized or "written by" in normalized:
        author = str(match.get("author") or "").strip()
        if author:
            return f"Author: {author}"
        return "The matched file does not have indexed author metadata."

    if "title" in normalized:
        title = str(match.get("title") or "").strip()
        if title:
            return f"Title: {title}"
        return f"Title: {file_name}"

    if "filename" in normalized or "file name" in normalized:
        return f"File name: {file_name}"

    if "type" in normalized or "format" in normalized or "extension" in normalized:
        return f"File type: {source_type}"

    if _is_explicit_file_location_query(normalized):
        return f"Matched file:\n{source_path}"

    return None


def _looks_like_identity_path(path: str) -> bool:
    return os.path.basename((path or "").strip()).lower() == _IDENTITY_FILE_NAME


def _is_self_identity_query(query: str) -> bool:
    normalized = _normalize_for_match(query)
    if not normalized:
        return False
    return any(hint in normalized for hint in _SELF_IDENTITY_QUERY_HINTS)


def _parse_identity_profile(text: str) -> dict:
    profile = {}
    current_section = None

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.endswith(":") and ":" not in line[:-1]:
            current_section = line[:-1].strip().lower().replace(" ", "_")
            profile.setdefault(current_section, [])
            continue

        if line.startswith("- "):
            if current_section:
                profile.setdefault(current_section, []).append(line[2:].strip())
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            profile[key.strip().lower().replace(" ", "_")] = value.strip()
            current_section = None

    return profile


def _build_identity_answer(query: str, identity_text: str) -> Optional[str]:
    if not _is_self_identity_query(query):
        return None

    profile = _parse_identity_profile(identity_text)
    name = profile.get("name") or "Nexus"
    version = profile.get("version")
    builder = profile.get("builder")
    mission = profile.get("core_mission") or profile.get("mission")
    capabilities = [item for item in profile.get("capabilities", []) if item]
    personality = [item for item in profile.get("personality", []) if item]

    capability_lines = [item for item in capabilities if item.lower().startswith("i can")]
    limitation_lines = [
        item
        for item in capabilities
        if item.lower().startswith("i cannot") or item.lower().startswith("i do not")
    ]

    normalized = _normalize_for_match(query)

    if "who built you" in normalized or "who made you" in normalized or "who created you" in normalized:
        if builder:
            return f"I was built by {builder}."
        return None

    if "mission" in normalized and mission:
        return mission

    if "version" in normalized and version:
        return f"I am {name}, version {version}."

    if "personality" in normalized and personality:
        return f"My personality is {', '.join(personality)}."

    if (
        "what can you do" in normalized
        or "what do you do" in normalized
        or "capabilities" in normalized
    ):
        if capability_lines:
            return "Here is what I can do:\n- " + "\n- ".join(capability_lines)

    if (
        "limitations" in normalized
        or "what cant you do" in normalized
        or "what can t you do" in normalized
        or "what can you not do" in normalized
    ):
        if limitation_lines:
            return "Here are my current limitations:\n- " + "\n- ".join(limitation_lines)

    if "offline" in normalized:
        offline_line = next((item for item in capabilities if "offline" in item.lower()), "")
        if offline_line:
            return offline_line

    if "private" in normalized or "personal data" in normalized:
        privacy_line = next(
            (
                item
                for item in capabilities
                if "personal data" in item.lower() or "cloud" in item.lower() or "privacy" in item.lower()
            ),
            "",
        )
        if privacy_line:
            return privacy_line

    intro_parts = [f"I'm {name}"]
    if version:
        intro_parts.append(f"version {version}")
    intro = ", ".join(intro_parts)
    if builder:
        intro += f", built by {builder}"
    intro += "."

    summary_parts = [intro]
    if mission:
        summary_parts.append(mission)
    if capability_lines:
        summary_parts.append("I can " + "; ".join(item[6:].rstrip(".") for item in capability_lines[:3]) + ".")
    if limitation_lines:
        summary_parts.append("Current limitation: " + limitation_lines[0])

    return " ".join(part.strip() for part in summary_parts if part.strip())


def _should_apply_lexical_supplement(query: str) -> bool:
    if _is_existence_or_list_query(query):
        return True
    query_terms = _query_terms(query)
    if not query_terms:
        return False

    text_lookup_terms = _NOTE_INTENT_TERMS | {"game", "games", "adventure", "story", "stories", "outline", "outlines"}
    return bool(query_terms.intersection(text_lookup_terms))


def _lexical_document_candidates(
    query: str,
    file_filter: str = "",
    limit: int = 50,
    workspace_id: str = "",
) -> list[dict]:
    query_terms = _query_terms(query)
    normalized_query = _normalize_for_match(query)
    if not query_terms:
        return []

    normalized_file_filter = (file_filter or "").strip()
    abs_filter = os.path.abspath(normalized_file_filter) if normalized_file_filter else None
    candidates = []

    for row in load_rows(Config.MULTIMODAL_DOCUMENTS_TABLE):
        source_path = os.path.abspath(row.get("source_path") or "")
        if not source_path:
            continue
        if abs_filter and source_path != abs_filter:
            continue
        if workspace_id and row.get("workspace_id") != workspace_id:
            continue

        haystack = " ".join(
            [
                str(row.get("file_name") or ""),
                str(row.get("title") or ""),
                source_path,
            ]
        )
        overlap = _text_relevance_overlap_score(query_terms, haystack)
        intent_bonus = _intent_source_bonus(row, query_terms)
        phrase_bonus = 0.0
        normalized_haystack = _normalize_for_match(haystack)
        if "text based adventure" in normalized_query and "text based adventure" in normalized_haystack:
            phrase_bonus += 0.8

        score = overlap + intent_bonus + phrase_bonus
        if score <= 0:
            continue

        candidate = dict(row)
        candidate["source_path"] = source_path
        candidate["_lexical_score"] = score
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            float(item.get("_lexical_score") or 0.0),
            str(item.get("file_name") or "").lower(),
        ),
        reverse=True,
    )
    return candidates[:limit]


def _merge_document_rows(primary_rows: list[dict], supplemental_rows: list[dict], limit: int = 100) -> list[dict]:
    merged = []
    seen_paths = set()

    for row in list(primary_rows) + list(supplemental_rows):
        source_path = os.path.abspath(row.get("source_path") or "")
        if not source_path or source_path in seen_paths:
            continue
        normalized = dict(row)
        normalized["source_path"] = source_path
        merged.append(normalized)
        seen_paths.add(source_path)
        if len(merged) >= limit:
            break
    return merged


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


def _plan_local_retrieval(query: str, file_filter: str = "", workspace_id: str = "") -> dict:
    context_line = (
        f"Focused file: {os.path.abspath(file_filter)}"
        if file_filter
        else "Focused file: none"
    )
    workspace_line = f"Workspace: {workspace_id}" if workspace_id else "Workspace: global"
    messages = [
        SystemMessage(content=_LOCAL_RETRIEVAL_PLAN_PROMPT),
        HumanMessage(content=f"{context_line}\n{workspace_line}\nUser query: {query}"),
    ]

    started_at = time.perf_counter()
    print(
        "local retrieval planner start | "
        f"mode=legacy | workspace_id={workspace_id or 'global'} | "
        f"query={query[:160]!r}"
    )
    try:
        response = _get_retrieval_planner().invoke(messages)
        plan = _normalize_retrieval_plan(_extract_json_object(getattr(response, "content", "") or ""))
    except Exception as exc:
        print(f"⚠️ Local retrieval planner failed: {exc}")
        plan = {"retrieval_mode": "semantic_search", "response_mode": "snippets"}
    finally:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
        print(f"local retrieval planner end | mode=legacy | elapsed_ms={elapsed_ms}")

    print(f"local retrieval plan: {plan}")
    return plan


def _plan_local_retrieval_v2(query: str, file_filter: str = "", workspace_id: str = "") -> str:
    context_line = (
        f"Focused file: {os.path.abspath(file_filter)}"
        if file_filter
        else "Focused file: none"
    )
    workspace_line = f"Workspace: {workspace_id}" if workspace_id else "Workspace: global"
    messages = [
        SystemMessage(content=_LOCAL_RETRIEVAL_V2_PROMPT),
        HumanMessage(content=f"{context_line}\n{workspace_line}\nUser query: {query}"),
    ]
    default_mode = "semantic_answer"
    started_at = time.perf_counter()
    print(
        "local retrieval planner start | "
        f"mode=v2 | workspace_id={workspace_id or 'global'} | "
        f"query={query[:160]!r}"
    )
    try:
        response = _get_retrieval_planner().invoke(messages)
        payload = _extract_json_object(getattr(response, "content", "") or "")
        mode = str(payload.get("mode", "")).strip().lower()
        if mode not in {"semantic_answer", "document_lookup", "catalog", "full_document", "hybrid"}:
            mode = default_mode
        return mode
    except Exception as exc:
        print(f"⚠️ Local retrieval planner v2 failed: {exc}")
        return default_mode
    finally:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
        print(f"local retrieval planner end | mode=v2 | elapsed_ms={elapsed_ms}")


def _compact_inventory_from_matches(matches: list[dict], limit: int = 12) -> str:
    if not matches:
        return "No matching indexed files were found."
    lines = [f"Matching files ({len(matches)}):"]
    for row in matches[:limit]:
        title = row.get("title") or row.get("file_name") or os.path.basename(row.get("source_path") or "")
        path = row.get("source_path") or ""
        lines.append(f"- {title} :: {path}")
    if len(matches) > limit:
        lines.append(f"... and {len(matches) - limit} more")
    return "\n".join(lines)


def _build_catalog_summary(matches: list[dict], workspace_id: str = "", limit: int = 30) -> str:
    if not matches:
        return (
            "No indexed files were found in the selected workspace."
            if workspace_id
            else "No indexed files were found."
        )

    header = (
        f"I found {len(matches)} indexed file(s) in the selected workspace:"
        if workspace_id
        else f"I found {len(matches)} indexed file(s):"
    )
    lines = [header]
    for row in matches[:limit]:
        source_path = os.path.abspath(row.get("source_path") or "")
        file_name = row.get("file_name") or os.path.basename(source_path)
        source_type = row.get("source_type") or _source_type_from_path(source_path)
        lines.append(f"- {file_name} ({source_type}) :: {source_path}")
    if len(matches) > limit:
        lines.append(f"... and {len(matches) - limit} more")
    return "\n".join(lines)


def _dedupe_source_metadata(items: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for item in items:
        key = (str(item.get("title", "")), str(item.get("url", "")), str(item.get("type", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


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


def get_nexus_identity_path() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    return os.path.join(project_root, "data", _IDENTITY_FILE_NAME)


def get_nexus_identity_response(query: str = "") -> tuple[str, list[dict]]:
    identity_path = get_nexus_identity_path()
    abs_path = os.path.abspath(identity_path)
    metadata = [{
        "title": _source_title(abs_path, "txt"),
        "url": abs_path,
        "type": "local",
    }]

    if not os.path.exists(abs_path):
        return (
            "The Nexus identity file is not available.",
            metadata,
        )

    identity_text = _load_full_text(abs_path, "txt")
    direct_answer = _build_identity_answer(query, identity_text)
    if direct_answer:
        return "", metadata + [build_final_response_artifact(direct_answer)]

    framing_header = (
        "The following content is Nexus's canonical self-profile. "
        "Answer the user's question using this profile only.\n"
        "─────────────────────────────────────────\n"
    )
    return framing_header + identity_text, metadata


def _query_documents_table(query: str, file_filter: str = "", workspace_id: str = "") -> list[dict]:
    normalized_file_filter = (file_filter or "").strip() or None
    started_at = time.perf_counter()
    print(
        "documents table query start | "
        f"workspace_id={workspace_id or 'global'} | file_filter={normalized_file_filter or 'none'} | "
        f"query={query[:160]!r}"
    )
    structured_query, _, _ = build_multimodal_structured_query(
        query,
        file_filter=normalized_file_filter,
        workspace_id=(workspace_id or "").strip() or None,
    )
    filter_node = getattr(structured_query, "filter", None)
    lexical_rows = _lexical_document_candidates(
        query,
        normalized_file_filter or "",
        limit=60,
        workspace_id=(workspace_id or "").strip(),
    )

    rows = []
    if filter_node is not None:
        for row in load_rows(Config.MULTIMODAL_DOCUMENTS_TABLE):
            source_path = os.path.abspath(row.get("source_path") or "")
            if not source_path:
                continue
            normalized_row = dict(row)
            normalized_row["source_path"] = source_path
            if _row_matches_filter(normalized_row, filter_node):
                rows.append(normalized_row)

    combined = _merge_document_rows(rows, lexical_rows, limit=100)
    sorted_rows = sorted(
        combined,
        key=lambda item: (
            float(item.get("_lexical_score") or 0.0),
            (item.get("file_name") or "").lower(),
            (item.get("source_path") or "").lower(),
        ),
        reverse=True,
    )
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    print(
        "documents table query end | "
        f"workspace_id={workspace_id or 'global'} | rows={len(sorted_rows)} | elapsed_ms={elapsed_ms}"
    )
    return sorted_rows


def _query_document_catalog(file_filter: str = "", workspace_id: str = "") -> list[dict]:
    normalized_file_filter = (file_filter or "").strip()
    absolute_filter = os.path.abspath(normalized_file_filter) if normalized_file_filter else ""
    normalized_workspace = (workspace_id or "").strip()
    rows = []

    for row in load_rows(Config.MULTIMODAL_DOCUMENTS_TABLE):
        source_path = os.path.abspath(row.get("source_path") or "")
        if not source_path:
            continue
        if absolute_filter and source_path != absolute_filter:
            continue
        row_workspace = (row.get("workspace_id") or "global").strip() or "global"
        if normalized_workspace:
            if row_workspace != normalized_workspace:
                continue
        elif row_workspace != "global":
            continue

        normalized_row = dict(row)
        normalized_row["source_path"] = source_path
        rows.append(normalized_row)

    return sorted(
        rows,
        key=lambda item: (
            (item.get("file_name") or "").lower(),
            (item.get("source_path") or "").lower(),
        ),
    )


def _build_existence_listing_response(matches: list[dict], query: str) -> str:
    if not matches:
        return "No matching indexed files were found."

    lines = [f"Yes, I found {len(matches)} matching idea file(s):"]
    for row in matches[:30]:
        label = row.get("title") or row.get("file_name") or os.path.basename(row.get("source_path") or "")
        lines.append(f"- {label}")
    if len(matches) > 30:
        lines.append(f"... and {len(matches) - 30} more")
    lines.append("")
    lines.append(f"Matched for: {query}")
    return "\n".join(lines)


def _supplement_with_lexical_parent_rows(
    rows: list[dict],
    query: str,
    file_filter: str = "",
    workspace_id: str = "",
    limit: int = 10,
) -> list[dict]:
    lexical_docs = _lexical_document_candidates(
        query,
        file_filter,
        limit=max(limit, 12),
        workspace_id=workspace_id,
    )
    if not lexical_docs:
        return rows

    existing_paths = {os.path.abspath(row.get("source_path") or "") for row in rows}
    parent_rows = load_rows(Config.MULTIMODAL_PARENT_TABLE)
    supplemental = []
    for doc in lexical_docs:
        source_path = os.path.abspath(doc.get("source_path") or "")
        if not source_path or source_path in existing_paths:
            continue

        parent_candidates = [
            row
            for row in parent_rows
            if os.path.abspath(row.get("source_path") or "") == source_path
            and (not workspace_id or row.get("workspace_id") == workspace_id)
            and row.get("modality", "text") == "text"
            and (row.get("text") or "").strip()
        ]
        if not parent_candidates:
            continue

        parent_candidates.sort(
            key=lambda item: (
                item.get("page") if isinstance(item.get("page"), int) else -1,
                item.get("parent_index") if isinstance(item.get("parent_index"), int) else -1,
            )
        )
        selected_parent = dict(parent_candidates[0])
        extra = parse_extra(selected_parent.get("extra"))
        if not extra.get("matched_text"):
            extra["matched_text"] = (selected_parent.get("text") or "")[:260]
        selected_parent["extra"] = json.dumps(extra, ensure_ascii=True)
        selected_parent["_score"] = float(doc.get("_lexical_score") or 0.0)
        supplemental.append(selected_parent)
        existing_paths.add(source_path)
        if len(supplemental) >= limit:
            break

    return list(rows) + supplemental


def _resolve_direct_local_response_legacy(
    query: str,
    file_filter: str = "",
    workspace_id: str = "",
) -> Optional[tuple[str, list[dict]]]:
    normalized_file_filter = (file_filter or "").strip()
    is_inventory_query = _is_explicit_document_inventory_query(query)
    wants_full_content = _wants_full_document(query)

    if not is_inventory_query and not wants_full_content and _is_content_extraction_query(query):
        return None

    matches = _query_documents_table(query, normalized_file_filter, workspace_id=workspace_id)
    metadata = _build_source_metadata(matches)

    if is_inventory_query:
        listing = _build_existence_listing_response(matches, query)
        return "", metadata + [build_final_response_artifact(listing)]

    if not matches:
        return (
            "",
            metadata
            + [
                build_final_response_artifact(
                    "No indexed files matched the requested document lookup in the selected workspace."
                    if workspace_id
                    else "No indexed files matched the requested document lookup."
                )
            ],
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
        if _looks_like_identity_path(source_path):
            identity_text = _load_full_text(source_path, (match.get("source_type") or _source_type_from_path(source_path)).lower())
            identity_answer = _build_identity_answer(query, identity_text)
            if identity_answer:
                return "", metadata + [build_final_response_artifact(identity_answer)]
        metadata_answer = _document_metadata_answer(query, match)
        if metadata_answer:
            return "", metadata + [build_final_response_artifact(metadata_answer)]
        return None

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


def _resolve_direct_local_response_v2(
    query: str,
    file_filter: str = "",
    workspace_id: str = "",
    explicit_mode: str = "document_lookup",
) -> tuple[str, list[dict]]:
    normalized_file_filter = (file_filter or "").strip()
    mode = (explicit_mode or "document_lookup").strip().lower()
    if mode not in {"document_lookup", "full_document", "hybrid"}:
        mode = "document_lookup"
    wants_full_content = mode == "full_document"

    matches = _query_documents_table(query, normalized_file_filter, workspace_id=workspace_id)
    metadata = _build_source_metadata(matches)
    if not matches:
        return (
            "",
            metadata
            + [
                build_final_response_artifact(
                    "No indexed files matched the requested document lookup in the selected workspace."
                    if workspace_id
                    else "No indexed files matched the requested document lookup."
                )
            ],
        )

    if wants_full_content and len(matches) > 1:
        listing = _compact_inventory_from_matches(matches, limit=20)
        message = f"{listing}\n\nSpecify the exact file to return full content."
        return "", metadata + [build_final_response_artifact(message)]

    if not wants_full_content and len(matches) > 1:
        listing = _compact_inventory_from_matches(matches, limit=20)
        return "", metadata + [build_final_response_artifact(listing)]

    match = matches[0]
    source_path = match["source_path"]
    source_type = (match.get("source_type") or _source_type_from_path(source_path)).lower()

    if not wants_full_content:
        if _looks_like_identity_path(source_path):
            identity_text = _load_full_text(source_path, source_type)
            identity_answer = _build_identity_answer(query, identity_text)
            if identity_answer:
                return "", metadata + [build_final_response_artifact(identity_answer)]
        metadata_answer = _document_metadata_answer(query, match)
        if metadata_answer:
            return "", metadata + [build_final_response_artifact(metadata_answer)]
        return "", metadata + [build_final_response_artifact(f"Matched file:\n{source_path}")]

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


def resolve_direct_local_response(
    query: str,
    file_filter: str = "",
    workspace_id: str = "",
) -> Optional[tuple[str, list[dict]]]:
    if Config.RAG_PIPELINE_VERSION == "manager_v2":
        mode = _plan_local_retrieval_v2(query, file_filter, workspace_id=workspace_id)
        if mode in {"document_lookup", "full_document", "hybrid"}:
            return _resolve_direct_local_response_v2(
                query,
                file_filter,
                workspace_id=workspace_id,
                explicit_mode=mode,
            )
        return None
    return _resolve_direct_local_response_legacy(query, file_filter, workspace_id=workspace_id)


def execute_local_retrieval_task_v2(
    query: str,
    file_filter: str = "",
    workspace_id: str = "",
    mode: str = "semantic_answer",
) -> dict:
    normalized_mode = (mode or "").strip().lower()
    if normalized_mode not in {"semantic_answer", "document_lookup", "catalog", "full_document", "hybrid"}:
        normalized_mode = _plan_local_retrieval_v2(query, file_filter, workspace_id=workspace_id)

    source_metadata: list[dict] = []
    evidence: list[dict] = []
    summary_parts: list[str] = []

    if normalized_mode == "catalog":
        matches = _query_document_catalog(file_filter=file_filter, workspace_id=workspace_id)
        source_metadata = _build_source_metadata(matches)
        summary = _build_catalog_summary(matches, workspace_id=workspace_id, limit=30)
        evidence = [
            {
                "source_type": "local",
                "title": _source_title(row.get("source_path") or "", row.get("source_type")),
                "url": os.path.abspath(row.get("source_path") or ""),
                "snippet": _trim_excerpt(
                    f"file_name={row.get('file_name') or os.path.basename(row.get('source_path') or '')}; "
                    f"source_type={row.get('source_type') or _source_type_from_path(row.get('source_path') or '')}; "
                    f"path={os.path.abspath(row.get('source_path') or '')}",
                    limit=240,
                ),
                "score": 1.0,
                "metadata": {"channel": "catalog"},
            }
            for row in matches[:30]
        ]
        return {
            "worker": "local_catalog_worker",
            "status": "ok" if matches else "empty",
            "summary": summary[:12000],
            "proposed_answer": summary[:4000],
            "evidence": evidence,
            "source_metadata": _dedupe_source_metadata(source_metadata),
        }

    if normalized_mode in {"semantic_answer", "hybrid"}:
        semantic_results = _search_multimodal_results(
            query,
            file_filter=(file_filter or "").strip() or None,
            workspace_id=(workspace_id or "").strip() or None,
            top_k=8,
            apply_lexical_supplement=False,
        )
        for index, result in enumerate(semantic_results[:8], start=1):
            snippet = (result.content or "").strip()
            summary_parts.append(f"[semantic {index}] {result.title}\n{snippet[:900]}")
            source_metadata.append({"title": result.title, "url": result.url, "type": "local"})
            evidence.append(
                {
                    "source_type": "local",
                    "title": result.title,
                    "url": result.url,
                    "snippet": snippet[:1200],
                    "score": max(0.0, 1.0 - (0.08 * (index - 1))),
                    "metadata": {"channel": "semantic"},
                }
            )

    if normalized_mode in {"document_lookup", "full_document", "hybrid"}:
        lookup_mode = "full_document" if normalized_mode == "full_document" else "document_lookup"
        lookup_payload = _resolve_direct_local_response_v2(
            query,
            file_filter,
            workspace_id=workspace_id,
            explicit_mode=lookup_mode,
        )
        lookup_text = extract_final_response(lookup_payload[1]) if lookup_payload else ""
        lookup_sources = [
            item
            for item in (lookup_payload[1] if lookup_payload else [])
            if isinstance(item, dict) and item.get("type") != "final_response"
        ]
        if lookup_text:
            heading = "[lookup]"
            if normalized_mode == "hybrid":
                heading = "[inventory]"
            summary_parts.append(f"{heading} {lookup_text}")
            evidence.append(
                {
                    "source_type": "local",
                    "title": "Local document lookup",
                    "url": "",
                    "snippet": lookup_text[:1800],
                    "score": 0.75,
                    "metadata": {"channel": "lookup", "mode": lookup_mode},
                }
            )
        source_metadata.extend(lookup_sources)

    source_metadata = _dedupe_source_metadata(source_metadata)
    status = "ok" if summary_parts else "empty"
    summary = "\n\n".join(summary_parts) if summary_parts else (
        "I couldn't find relevant information in the selected workspace."
        if workspace_id
        else "No relevant local evidence found."
    )
    proposed_answer = summary if normalized_mode in {"document_lookup", "full_document"} else ""
    return {
        "worker": "local_retrieval_worker",
        "status": status,
        "summary": summary[:12000],
        "proposed_answer": proposed_answer[:4000],
        "evidence": evidence,
        "source_metadata": source_metadata,
    }


def _search_local_legacy(query: str, file_filter: str = "", workspace_id: str = "") -> List[SearchResult]:
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
        multimodal_results = _search_multimodal_results(
            query,
            file_filter=file_filter,
            workspace_id=(workspace_id or "").strip() or None,
            top_k=top_k,
            apply_lexical_supplement=True,
        )
    except Exception as exc:
        print(f"⚠️ Multimodal search failed: {exc}")
        multimodal_results = []

    if not multimodal_results:
        msg = "No relevant local documents found."
        if file_filter:
            msg = f"The focused document '{os.path.basename(file_filter)}' does not seem to contain information regarding your query."
        elif workspace_id:
            msg = "I couldn't find relevant information in the selected workspace."
        return [SearchResult(title="Info", url="", content=msg)]

    return multimodal_results


def _search_local_v2(query: str, file_filter: str = "", workspace_id: str = "") -> List[SearchResult]:
    if not Config.MULTIMODAL_EMBEDDINGS_ENABLED:
        return [SearchResult(title="System", url="local", content="No local knowledge found. Please ingest files or folders first.")]

    normalized_file_filter = (file_filter or "").strip()
    mode = _plan_local_retrieval_v2(query, normalized_file_filter, workspace_id=workspace_id)
    payload = execute_local_retrieval_task_v2(
        query=query,
        file_filter=normalized_file_filter,
        workspace_id=workspace_id,
        mode=mode,
    )

    results: list[SearchResult] = []
    for evidence in payload.get("evidence", []):
        if not isinstance(evidence, dict):
            continue
        results.append(
            SearchResult(
                title=evidence.get("title", "Local Evidence"),
                url=evidence.get("url", ""),
                content=evidence.get("snippet", ""),
                source="local",
            )
        )

    if results:
        return results[:10]

    fallback = (payload.get("summary") or "").strip() or "No relevant local evidence found."
    return [SearchResult(title="Info", url="", content=fallback, source="local")]


def search_local(query: str, file_filter: str = "", workspace_id: str = "") -> List[SearchResult]:
    if Config.RAG_PIPELINE_VERSION == "manager_v2":
        return _search_local_v2(query, file_filter=file_filter, workspace_id=workspace_id)
    return _search_local_legacy(query, file_filter=file_filter, workspace_id=workspace_id)


def _search_multimodal_results(
    query: str,
    file_filter: str = None,
    workspace_id: str = None,
    top_k: int = 10,
    apply_lexical_supplement: bool = True,
) -> List[SearchResult]:
    rows = search_multimodal(query, top_k=top_k, file_filter=file_filter, workspace_id=workspace_id) or []
    semantic_query = query
    if rows:
        semantic_query = str(rows[0].get("_semantic_query") or query)
        if semantic_query.strip() != (query or "").strip():
            print(f"local rerank query | original={query!r} | effective={semantic_query!r}")

    if apply_lexical_supplement and _should_apply_lexical_supplement(semantic_query):
        rows = _supplement_with_lexical_parent_rows(
            rows,
            query=semantic_query,
            file_filter=file_filter or "",
            workspace_id=workspace_id or "",
            limit=min(max(top_k, 4), 12),
        )
    if not rows:
        return []
    # Keep context focused by reranking and deduplicating per source file.
    rows = _rerank_and_filter_rows(rows, query=semantic_query, top_k=min(top_k, 5))

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
