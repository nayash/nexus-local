from langchain_ollama import OllamaEmbeddings
from src.core.config import Config
from src.rag.storage import get_table, list_tables
from src.tools.schemas import SearchResult
from src.rag.ingestion import NexusIngestor
from typing import List
import os

# Initialize embeddings_model (keep global if needed elsewhere, but we'll use Ingestor)
embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=Config.OLLAMA_BASE_URL
)

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

    tables_to_search = []
    if file_filter:
         tables_to_search = ["documents"]
    else:
         tables_to_search = list_tables()
         
    if not tables_to_search:
        return [SearchResult(title="System", url="local", content="No local knowledge found. Please ingest files or folders first.")]

    # Initialize Ingestor (Strategy should ideally be loaded from config, but we default to parent)
    # Using 'parent' strategy allows it to handle both parent-child and fallback to naive if needed
    ingestor = NexusIngestor(strategy="parent")
    
    all_results = []
    
    for table_name in tables_to_search:
        # result_limit=20 for focused, 10 for global
        result_limit = 20 if file_filter else 10
        
        try:
            # NexusIngestor.search handles the strategy logic (Naive vs Parent)
            # and returns Document objects.
            print(f"--- 🔍 SEARCHING TABLE: {table_name} ---")
            docs = ingestor.search(query, k=result_limit, table_name=table_name)
            
            for doc in docs:
                # Add metadata for tracking
                metadata = doc.metadata or {}
                source = metadata.get("source", "Unknown Source")
                
                # Apply file filter if set (though ideally search() should handle it)
                if file_filter and os.path.abspath(source) != os.path.abspath(file_filter):
                    continue
                
                all_results.append(
                    SearchResult(
                        title=f'Local File ({table_name}): {os.path.basename(source)}',
                        url=source,
                        content=doc.page_content,
                        source="local"
                    )
                )
                print(f'added search result: {all_results[-1]}')
        except Exception as e:
            print(f"Error searching table {table_name}: {e}")

    if not all_results:
        msg = "No relevant local documents found."
        if file_filter:
            msg = f"The focused document '{os.path.basename(file_filter)}' does not seem to contain information regarding your query."
        return [SearchResult(title="Info", url="", content=msg)]
        
    return all_results