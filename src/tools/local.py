from langchain_ollama import OllamaEmbeddings
from src.core.config import Config
from src.rag.storage import get_table
from src.tools.schemas import SearchResult
from typing import List
import os

embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=Config.OLLAMA_BASE_URL
)

def search_local(query: str, file_filter: str = None) -> List[SearchResult]:
    """
    Searches the local LanceDB for relevant document chunks.
    Args:
        query: The search text.
        file_filter: Optional absolute path to restrict search to a specific file.
    """
    tbl = get_table("documents")
    if not tbl:
        return [SearchResult(title="System", url="local", content="No local documents found. Please ingest files first.")]

    # 1. Embed the query
    query_vector = embeddings_model.embed_query(query)

    # 2. Search DB (Vector Search)
    # limit=3 for global search, limit=15 for focused search to provide deep context
    result_limit = 15 if file_filter else 3
    search_builder = tbl.search(query_vector).limit(result_limit)
    
    if file_filter:
        print(f"--- 🎯 FOCUSED SEARCH: '{file_filter}' ---")
        search_builder = search_builder.where(f"source = '{file_filter}'")
        
    results = search_builder.to_list()
    
    filtered_results = []
    # THRESHOLD CONFIG
    # In LanceDB/L2 Distance: LOWER score is BETTER (0 = exact match)
    # A score > 0.4 usually implies "vaguely related but not relevant"
    MAX_DISTANCE = 0.8

    for r in results:
        score = r.get("_distance", 1.0)
        
        # DEBUG: Print scores to tune this value
        print(f"DEBUG: Found '{r['source']}' with distance: {score}")
        
        # ROBUSTNESS FIX: If user is focusing on a specific file, we skip the distance check.
        # We assume the user knows this file is relevant.
        if file_filter or score <= MAX_DISTANCE:
            # Only keep good matches (or all if focused)
            filtered_results.append(
                SearchResult(
                    title=f'Local File: {os.path.basename(r["source"])}',
                    url=r["source"], 
                    content=r["text"],
                    source="local"
                )
            )
    
    if not filtered_results:
        msg = "No relevant local documents found."
        if file_filter:
            msg = f"The focused document '{os.path.basename(file_filter)}' does not seem to contain information regarding your query."
        return [SearchResult(title="Info", url="", content=msg)]
        
    return filtered_results