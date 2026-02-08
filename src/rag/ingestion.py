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
from langchain_classic.storage import EncoderBackedStore
from langchain_core.documents import Document
import pickle

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
            # LocalFileStore stores bytes, so we wrap it with EncoderBackedStore to store Documents via pickle
            self.fs_store = LocalFileStore(self.docstore_path)
            self.docstore = EncoderBackedStore(
                store=self.fs_store,
                key_encoder=lambda x: x, # Simple identity for keys
                value_serializer=pickle.dumps,
                value_deserializer=pickle.loads
            )
            
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
            return self._ingest_parent(file_path, table_name)
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
            # Support search on specific tables with parent strategy
            if table_name and table_name != self.vector_table_name:
                 try:
                     vstore = LanceDB(
                        connection=self.db,
                        embedding=embeddings_model,
                        table_name=table_name
                     )
                     temp_retriever = ParentDocumentRetriever(
                        vectorstore=vstore,
                        docstore=self.docstore,
                        child_splitter=self.child_splitter,
                        parent_splitter=self.parent_splitter
                     )
                     return temp_retriever.invoke(query)[:k]
                 except Exception:
                     # Fallback or empty if table doesn't exist
                     return []

            return self.retriever.invoke(query)[:k]


    def ingest_directory(self, path: str, recursive: bool = True, table_name: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
        """
        Ingests all supported files from a directory.
        """
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(path):
            return False, "Path does not exist.", None
            
        if not os.path.isdir(path):
            return False, "Path is not a directory.", None

        # Sanitize table name for the folder if not provided
        if not table_name:
            table_name = self._sanitize_table_name(os.path.basename(path))
            
        print(f"--- 📂 DIRECTORY DETECTED: Will ingest into table '{table_name}' ---")
        
        files_to_ingest = self._find_files(path, recursive)
        
        if not files_to_ingest:
            return False, "No supported files found.", None
            
        print(f"Total files to ingest: {len(files_to_ingest)},\n,{files_to_ingest}\n")
        
        total_chunks = 0
        successful_files = 0
        
        for file_path in files_to_ingest:
            success, chunks, _ = self.ingest_file(file_path, table_name=table_name)
            if success:
                successful_files += 1
                total_chunks += chunks
                
        msg = f"Successfully ingested {successful_files} files ({total_chunks} chunks) into '{table_name}'."
        return True, msg, None

    def _find_files(self, path: str, recursive: bool) -> List[str]:
        """
        Finds all supported files in a directory.
        """
        supported_extensions = (".pdf", ".txt", ".md", ".csv", ".sh")
        files_found = []
        
        if recursive:
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(supported_extensions):
                        files_found.append(os.path.join(root, file))
        else:
            for file in os.listdir(path):
                if file.endswith(supported_extensions):
                    files_found.append(os.path.join(path, file))
                    
        return files_found
        
    def _sanitize_table_name(self, name: str) -> str:
        """
        Sanitizes a string to be a valid table name.
        """
        sanitized = "folder_" + name.strip().replace(" ", "_").replace("-", "_").replace(".", "").lower()
        return sanitized[:63]

    def _ingest_parent(self, file_path: str, table_name: str = None):
        try:
            target_table = table_name if table_name else self.vector_table_name
            print(f"--- 📥 INGESTING (Parent-Child): {file_path} into '{target_table}' ---")
            
            # 1. Load
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                # Enable autodetect_encoding to handle various text encodings
                loader = TextLoader(file_path, autodetect_encoding=True)
            docs = loader.load()
            
            # 2. Prepare Retriever
            target_retriever = self.retriever
            
            # If table_name is specified and different from default, create a temporary retriever
            if table_name and table_name != self.vector_table_name:
                vstore = LanceDB(
                    connection=self.db,
                    embedding=embeddings_model,
                    table_name=table_name,
                    mode="append"
                )
                # Reuse the same docstore (filesystem) but different vector table
                target_retriever = ParentDocumentRetriever(
                    vectorstore=vstore,
                    docstore=self.docstore,
                    child_splitter=self.child_splitter,
                    parent_splitter=self.parent_splitter
                )
            
            # 3. Add to Retriever (Handles splitting & storing)
            
            # SANITIZATION: Convert metadata fields to simple types for LanceDB compatibility
            # LanceDB strict schema enforcement can fail on complex types or inconsistent fields.
            for doc in docs:
                keys_to_remove = []
                for key, value in doc.metadata.items():
                    # 1. Remove None values (can cause schema issues)
                    if value is None:
                        keys_to_remove.append(key)
                        continue
                    
                    # 2. Convert complex types to strings (e.g. dates) to preserve data without breaking schema
                    # We keep basic types (str, int, float, bool) and stringify everything else
                    if not isinstance(value, (str, int, float, bool)):
                        doc.metadata[key] = str(value)

                for key in keys_to_remove:
                    doc.metadata.pop(key, None)
                    
                # Ensure source is absolute path
                doc.metadata["source"] = os.path.abspath(file_path)

            target_retriever.add_documents(docs)
            
            print(f"✅ Success! Ingested {len(docs)} parent docs into '{target_table}'.")
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

def ingest_file(file_path: str, table_name: str = "documents", strategy: Literal["naive", "parent"] = "parent"):
    """
    Wrapper for existing code calls. Uses NAIVE strategy by default to match legacy behavior.
    """
    ingestor = NexusIngestor(strategy=strategy)
    return ingestor.ingest_file(file_path, table_name)

def ingest_path(path: str, strategy: Literal["naive", "parent"] = "naive"):
    """
    Ingests all files from a directory or a single file. (Naive strategy default)
    """
    print(f'ingesting path {path} with strategy {strategy}')
    
    path = os.path.abspath(os.path.expanduser(path))
    ingestor = NexusIngestor(strategy=strategy)
    
    if os.path.isfile(path):
        return ingestor.ingest_file(path)
    elif os.path.isdir(path):
        return ingestor.ingest_directory(path)
    else:
        return False, "Path does not exist.", None

# Quick test block
def init_knowledge():
    # Resolve path relative to this file to ensure it works when packaged
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up 2 levels: src/rag -> src -> project_root
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    data_file = os.path.join(project_root, "data", "nexus-identity.txt")
    
    ingest_file(data_file)