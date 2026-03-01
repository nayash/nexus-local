from src.core.config import Config
from src.tools.schemas import SearchResult
from src.rag.ingestion_multimodal import search_multimodal
from typing import List
import os

def search_local(query: str, file_filter: str = None) -> List[SearchResult]:
    """
    Searches the local LanceDB for relevant document chunks using the active strategy.
    Args:
        query: The search text.
        file_filter: Optional absolute path to restrict search to a specific file.
    """
    
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
            content = row.get("text") or ""
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
