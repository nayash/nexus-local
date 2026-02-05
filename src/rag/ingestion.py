import os
import shutil
from typing import List, Literal, Optional, Tuple, Any
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_classic.retrievers import ParentDocumentRetriever
# from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document

from src.core.config import Config
from src.rag.storage import get_db_connection, get_table

# Initialize the lightweight embedding model
embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=Config.OLLAMA_BASE_URL
)

class NexusIngestor:
    """
    Unified ingestion and retrieval engine supporting multiple strategies.
    Strategies:
    - 'naive': Simple chunking and vector search (legacy/default).
    - 'parent': Parent Document Retrieval (Small-to-Big).
    """

    def __init__(self, strategy: Literal["naive", "parent"] = "parent"):
        self.strategy = strategy
        self.docstore_path = os.path.join(Config.PROJECT_ROOT, "data", "docstore")
        
        # Ensure docstore directory exists
        if not os.path.exists(self.docstore_path):
            os.makedirs(self.docstore_path)

        # Initialize components for Parent Document Retrieval
        if self.strategy == "parent":
            self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            
            # Persistent storage for parent documents
            self.docstore = LocalFileStore(self.docstore_path)
            
            # Vector store for child chunks
            # We use a dedicated table for parent-child strategy
            self.vector_table_name = "nexus_parent_child"
            self.db = get_db_connection()
            
            # Create/Get table wrapper for LangChain
            # Note: LanceDB wrapper handles table creation if passed a table-like object or correct arguments
            # We pass the table object from our storage utility
            try:
                table = get_table(self.vector_table_name)
                if table is None:
                    # If table doesn't exist, we let the wrapper create it implicitly on add_texts
                    # But LangChain's LanceDB requires an opened table or None to start
                    # We will rely on LanceDB wrapper to create it.
                   pass
            except Exception:
                pass

            self.vectorstore = LanceDB(
                connection=self.db,
                embedding=embeddings_model,
                table_name=self.vector_table_name
            )

            self.retriever = ParentDocumentRetriever(
                vectorstore=self.vectorstore,
                docstore=self.docstore,
                child_splitter=self.child_splitter,
                parent_splitter=self.parent_splitter
            )

    def ingest_file(self, file_path: str, table_name: str = "documents"):
        """
        Ingests a file using the selected strategy.
        """
        if self.strategy == "naive":
            return self._ingest_naive(file_path, table_name)
        elif self.strategy == "parent":
            return self._ingest_parent(file_path)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def search(self, query: str, k: int = 5, table_name: str = "documents"):
        """
        Searches using the selected strategy.
        Returns a list of documents or chunks.
        """
        if self.strategy == "naive":
            # Naive search implementation
            # Note: The original naive search logic was in src/tools/local.py, not here.
            # But the user asked to put search here.
            # I will implement a basic naive search here that mimics what local.py does.
            return self._search_naive(query, k, table_name)
        elif self.strategy == "parent":
            return self.retriever.invoke(query)[:k]

    def _ingest_parent(self, file_path: str):
        try:
            print(f"--- 📥 INGESTING (Parent-Child): {file_path} ---")
            
            # 1. Load
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)
            docs = loader.load()
            
            # 2. Add to Retriever (Handles splitting & storing)
            self.retriever.add_documents(docs)
            
            print(f"✅ Success! Ingested {len(docs)} parent docs into 'nexus_parent_child'.")
            return True, len(docs), os.path.abspath(file_path)
            
        except Exception as e:
            print(f"   ❌ Error ingesting {file_path}: {e}")
            return False, str(e), None

    def _ingest_naive(self, file_path: str, table_name: str = "documents"):
        """
        Legacy naive ingestion logic.
        """
        try:
            print(f"--- 📥 INGESTING (Naive): {file_path} INTO TABLE: {table_name} ---")
            
            # 1. Load the file based on extension
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)
                
            docs = loader.load()
            print(f"   -> Loaded {len(docs)} pages/documents.")

            # 2. Split into chunks (Naive)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(docs)
            print(f"   -> Split into {len(chunks)} chunks.")

            # 3. Embed & Store
            # We prepare the data for LanceDB (List of dicts)
            data = []
            print("   -> Generating Embeddings (This uses GPU)...")
            
            for i, chunk in enumerate(chunks):
                # Generate vector (768 dimensions for nomic)
                vector = embeddings_model.embed_query(chunk.page_content)
                
                data.append({
                    "vector": vector,
                    "text": chunk.page_content,
                    "source": os.path.abspath(file_path),
                    "id": f"{os.path.basename(file_path)}_{i}_{hash(chunk.page_content)}"
                })

            # 4. Save to DB
            db = get_db_connection()
            
            try:
                tbl = get_table(table_name)
                if tbl:
                    # DEDUPLICATION LOGIC: Delete existing chunks for this file
                    abs_path = os.path.abspath(file_path)
                    print(f"   -> Cleaning existing vectors for: {abs_path}")
                    tbl.delete(f"source = '{abs_path}'")
                    
                    tbl.add(data)
                    print(f"   -> Updated '{table_name}' table with new vectors.")
                else:
                    # Table doesn't exist, create it
                    db.create_table(table_name, data)
                    print(f"   -> Created new table '{table_name}' with initial vectors.")

            except (ValueError, FileNotFoundError, NameError, Exception) as e:
                # Fallback for creation
                 try:
                    db.create_table(table_name, data)
                 except Exception:
                     # If it failed because it exists now or something
                     tbl = db.open_table(table_name)
                     tbl.add(data)
                
            print(f"✅ Success! Added {len(data)} vectors to {table_name}.")
            return True, len(data), os.path.abspath(file_path)

        except Exception as e:
            print(f"   ❌ Error ingesting {file_path}: {e}")
            return False, str(e), None

    def _search_naive(self, query: str, k: int = 5, table_name: str = "documents"):
        """
        Implementation of naive search internally.
        """
        tbl = get_table(table_name)
        if not tbl:
            return []
        
        query_vector = embeddings_model.embed_query(query)
        results = tbl.search(query_vector).limit(k).to_list()
        
        # Convert to Document format for consistency
        docs = []
        for r in results:
            docs.append(Document(
                page_content=r["text"],
                metadata={"source": r["source"], "id": r["id"]}
            ))
        return docs

# --- Backward Compatibility Wrappers ---

def ingest_file(file_path: str, table_name: str = "documents", strategy: Literal["naive", "parent"] = "naive"):
    """
    Wrapper for existing code calls. Uses NAIVE strategy by default to match legacy behavior.
    """
    ingestor = NexusIngestor(strategy=strategy)
    return ingestor.ingest_file(file_path, table_name)

def ingest_path(path: str, strategy: Literal["naive", "parent"] = "naive"):
    """
    Ingests all files from a directory or a single file. (Naive strategy default)
    """
    # Logic copied and adapted to use NexusIngestor
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    
    if not os.path.exists(path):
        return False, "Path does not exist.", None

    files_to_ingest = []
    table_name = "documents" # Default for single files

    if os.path.isfile(path):
        if path.endswith((".pdf", ".txt", ".md", ".csv", ".sh")):
            files_to_ingest.append(path)
            
    elif os.path.isdir(path):
        # Create a sanitized table name for the folder
        sanitized_name = "folder_" + path.strip(os.sep).replace(os.sep, "_").replace(".", "").replace("-", "_").replace(" ", "_").lower()
        table_name = sanitized_name[:63] 
        print(f"--- 📂 DIRECTORY DETECTED: Will ingest into table '{table_name}' ---")
        
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith((".pdf", ".txt", ".md", ".csv", ".sh")):
                    files_to_ingest.append(os.path.join(root, file))
    
    if not files_to_ingest:
        return False, "No supported files found (.pdf, .txt, .md, .csv, .sh).", None

    total_chunks = 0
    successful_files = 0
    
    ingestor = NexusIngestor(strategy=strategy) # Use strategy passed in argument
    
    for file_path in files_to_ingest:
        # Pass the determined table name
        success, chunks, _ = ingestor.ingest_file(file_path, table_name=table_name)
        if success:
            successful_files += 1
            total_chunks += chunks
            
    # For Focus Mode, if we ingested exactly one file, return its absolute path
    final_path = os.path.abspath(files_to_ingest[0]) if len(files_to_ingest) == 1 else None
    
    msg = f"Successfully ingested {successful_files} files ({total_chunks} chunks) into '{table_name}'."
    return True, msg, final_path

# Quick test block
def init_knowledge():
    # Resolve path relative to this file to ensure it works when packaged
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up 2 levels: src/rag -> src -> project_root
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    data_file = os.path.join(project_root, "data", "nexus-identity.txt")
    
    ingest_file(data_file)