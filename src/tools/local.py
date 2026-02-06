from langchain_ollama import OllamaEmbeddings
from src.core.config import Config
from src.rag.storage import get_table, list_tables
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
    
    # 1. Determine which tables to search
    
    # Validation: Ensure focused file actually exists
    if file_filter and not os.path.exists(file_filter):
        print(f"⚠️ WARNING: Intent to focus on '{file_filter}' ignored because file does not exist.")
        file_filter = None

    tables_to_search = []
    
    # If focused file, we typically assume it is in the 'documents' table
    # (since single file ingest goes there). 
    # BUT, if it was part of a folder ingest, it might be elsewhere.
    # For MVP simplicity, based on our rules:
    # - Single files -> 'documents'
    # - Folder files -> 'folder_xxx'
    # We will search ALL tables if file_filter is set too, but filter by source.
    # Or optimize: assume user manually attached it -> it went to 'documents'.
    # Let's stick to the prompt requirement: 
    # "If user clicks on attachment icon ... ingest in 'documents' table ... and answer from document"
    # So focused file = 'documents' table.
    
    if file_filter:
         tables_to_search = ["documents"]
    else:
         tables_to_search = list_tables()
         
    if not tables_to_search:
        return [SearchResult(title="System", url="local", content="No local knowledge found. Please ingest files or folders first.")]

    # 2. Embed the query
    query_vector = embeddings_model.embed_query(query)
    
    all_results = []
    
    for table_name in tables_to_search:
        tbl = get_table(table_name)
        if not tbl:
            continue
            
        # limit=3 for global search per table (we will aggregate), limit=10 for focused
        result_limit = 10 if file_filter else 5 
        
        search_builder = tbl.search(query_vector).limit(result_limit)
        
        if file_filter:
            print(f"--- 🎯 FOCUSED SEARCH in {table_name}: '{file_filter}' ---")
            search_builder = search_builder.where(f"source = '{file_filter}'")
            
        try:
             results = search_builder.to_list()
             for r in results:
                r["_table"] = table_name # Track origin
                all_results.append(r)
        except Exception as e:
            print(f"Error searching table {table_name}: {e}")

    # 3. Sort & Filter Aggregate Results
    # Sort by distance (ASC)
    all_results.sort(key=lambda x: x.get("_distance", 1.0))
    
    filtered_results = []
    # THRESHOLD CONFIG
    # In LanceDB/L2 Distance: LOWER score is BETTER (0 = exact match)
    # A score > 0.4 usually implies "vaguely related but not relevant"
    MAX_DISTANCE = 0.9 # earlier value was 0.8
    global_limit = 15 if file_filter else 5 # Max total chunks to return
    
    for r in all_results[:global_limit]:
        score = r.get("_distance", 1.0)
        
        # Normalize Result Fields (Handle different schemas from different strategies)
        # Parent Strategy: source is inside 'metadata' dict
        # Legacy Strategy: source is top-level key
        
        source = r.get("source")
        text = r.get("text")
        
        # Check metadata if source/text are missing (Parent Strategy Schema)
        if not source and "metadata" in r and isinstance(r["metadata"], dict):
            source = r["metadata"].get("source")
            
        # Fallback if text is missing but page_content exists
        if not text and "page_content" in r:
             text = r["page_content"]
             
        # Fallback for source
        if not source:
             source = "Unknown Source"

        print(f"DEBUG: Found '{source}' in table '{r.get('_table')}' with distance: {score}")

        if file_filter or score <= MAX_DISTANCE:
            filtered_results.append(
                SearchResult(
                    title=f'Local File ({r.get("_table", "doc")}): {os.path.basename(source)}',
                    url=source, 
                    content=text or "No content available",
                    source="local"
                )
            )
    
    if not filtered_results:
        msg = "No relevant local documents found."
        if file_filter:
            msg = f"The focused document '{os.path.basename(file_filter)}' does not seem to contain information regarding your query."
        return [SearchResult(title="Info", url="", content=msg)]
        
    return filtered_results