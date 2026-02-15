import os
import shutil
from typing import List, Literal, Optional, Tuple, Any
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_classic.retrievers import ParentDocumentRetriever
# from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_community.vectorstores import LanceDB
from langchain_classic.storage import EncoderBackedStore
from langchain_core.documents import Document
import pickle
from typing import Callable


import pwd
from typing import List, Tuple, Optional, Literal, Dict, Any

from src.core.config import Config
from src.core.user_settings import get_setting
from src.rag.storage import get_db_connection, get_table, ensure_table_has_core_fields

from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.base import load_query_constructor_runnable

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

        # Initialize LLM for Self-Query and other tasks
        self.llm = ChatOllama(
            model=get_setting("model_name", "llama3.1"),
            temperature=0,
            base_url=Config.OLLAMA_BASE_URL
        )

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

    def ingest_file(self, file_path: str, table_name: str = "documents", progress_callback: Optional[Callable] = None):
        """
        Ingests a file using the selected strategy.
        """
        if self.strategy == "naive":
            result = self._ingest_naive(file_path, table_name)
        elif self.strategy == "parent":
            result = self._ingest_parent(file_path, table_name)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
            
        if progress_callback:
            progress_callback()
            
        return result

    def search(self, query: str, k: int = 5, table_name: str = "documents"):
        """
        Searches using the selected strategy.
        Returns a list of documents or chunks.
        """
        if self.strategy == "naive":
            return self._search_naive(query, k, table_name)
        elif self.strategy == "parent":
            # Unified self-query search for ALL tables
            return self._search_with_filter(query, k, table_name)

    def _search_with_filter(self, query: str, k: int = 5, table_name: str = None) -> List[Document]:
        """
        Enhanced search with metadata filtering using SelfQueryRetriever logic.
        Works for ANY table - reuses main retriever when possible, creates temporary otherwise.
        """
        try:
            # 1. Determine target table (default to main vector table)
            target_table = table_name or self.vector_table_name
            
            # 2. Get or create retriever and vectorstore
            if target_table == self.vector_table_name:
                # Reuse pre-initialized retriever and vectorstore for main table
                retriever = self.retriever
                vectorstore = self.vectorstore
            else:
                # Create temporary retriever for custom table
                vectorstore = self._get_or_create_vectorstore(target_table)
                retriever = ParentDocumentRetriever(
                    vectorstore=vectorstore,
                    docstore=self.docstore,
                    child_splitter=self.child_splitter,
                    parent_splitter=self.parent_splitter
                )
            
            # 3. Define Metadata Schema
            metadata_field_info = [
                AttributeInfo(
                    name="author",
                    description="The author or owner of the file (e.g. 'alex', 'root').",
                    type="string",
                ),
                AttributeInfo(
                    name="title",
                    description="The title of the document.",
                    type="string",
                ),
                AttributeInfo(
                    name="source",
                    description="The file path source of the document.",
                    type="string",
                ),
                AttributeInfo(
                    name="year",
                    description="The year of creation.",
                    type="integer",
                ),
                AttributeInfo(
                    name="page",
                    description="The page number.",
                    type="integer",
                ),
            ]
            
            document_content_description = "A collection of user's personal files, code, documents, and reports."

        # 4. Construct Query Compiler
        # Use a custom prompt to help local LLMs understand filename filtering
        from langchain_core.prompts import ChatPromptTemplate
        
        examples = [
            ("Summarize the file 'GAN paper.pdf'", 
             {"query": "Summarize the file", "filter": "eq('title', 'GAN paper.pdf')"}),
            ("What does report_v1.txt say?", 
             {"query": "What does it say?", "filter": "eq('title', 'report_v1.txt')"}),
            ("Find documents by author 'Alice'", 
             {"query": "Find documents", "filter": "eq('author', 'Alice')"}),
            ("search in 'project_plan.md' for 'timeline'",
             {"query": "timeline", "filter": "eq('source', 'project_plan.md')"})
        ]
        
        # We use a Chat Model (LLM) to parse natural language queries into structured filters.
        query_constructor = load_query_constructor_runnable(
            llm=self.llm,
            document_contents=document_content_description,
            attribute_info=metadata_field_info,
            examples=examples # Pass examples to guide the LLM
        )

        # 5. Parse Query into structured format
        structured_query = query_constructor.invoke({"query": query})

        # --- FALLBACK HEURISTIC ---
        # If LLM failed to extract a filter but query looks like it has a filename
        if not structured_query.filter:
            import re
            from langchain_core.structured_query import Comparison, Comparator, Operation, Operator
            
            # Look for filenames like "something.ext"
            # Avoiding common abbreviations like "vs." "e.g." is hard but let's try strict extension matching
            filename_match = re.search(r'\b([\w\-. ]+\.(pdf|txt|md|csv|sh|py|docx))\b', query, re.IGNORECASE)
            
            if filename_match:
                filename = filename_match.group(1)
                print(f"--- ⚠️ SELF-QUERY FALLBACK: Detected filename '{filename}' in query. Forcing filter. ---")
                
                # Construct a filter: title == filename OR source contains filename (approximated as title for now)
                # Ideally we check both but let's stick to title for simplicity as per schema
                structured_query.filter = Comparison(comparator=Comparator.EQ, attribute='title', value=filename)

        print(f"--- USER QUERY: {query} 📝 STRUCTURED QUERY: {structured_query} ---")
        print(f"    Target Table: {target_table}")
        
        # 6. Check if filter exists and apply if needed
        # structured_query is a StructuredQuery object with 'filter' attribute
        if structured_query.filter:
            print(f"--- 🔍 DETECTED FILTER: {structured_query.filter} ---")
            
            # 7. Translate to VectorStore filter format
            # LanceDB (via LangChain) uses specific filter format (SQL-like usually)
            from langchain.retrievers.self_query.lancedb import LanceDBTranslator
            translator = LanceDBTranslator()
            
            # Visit to get native filter (kwargs)
            filter_kwargs = translator.visit_structured_query(structured_query)
            
            # If filter_kwargs is a tuple, extract filter
            native_filter = filter_kwargs[0] if isinstance(filter_kwargs, tuple) else filter_kwargs
            
            print(f"Applying LanceDB Filter: {native_filter}")
            
            # 8. Execute filtered search on the target vectorstore
            # Search children chunks with metadata filter, then map to parents
            children = vectorstore.similarity_search(
                query, 
                k=k*2, # Fetch more children to ensure we get enough parents
                filter=native_filter # 'filter' kwarg for LanceDB
            )
            
            # Map to parents
            parent_ids = set()
            final_docs = []
            for child in children:
                parent_id = child.metadata.get("doc_id") 
                if parent_id and parent_id not in parent_ids:
                    try:
                        parent = self.docstore.mget([parent_id])[0]
                        if parent:
                            final_docs.append(parent)
                            parent_ids.add(parent_id)
                    except Exception:
                        pass
                        
                    if len(final_docs) >= k:
                        break
                        
            return final_docs
            
        else:
            # No filter detected, use standard semantic search
            return retriever.invoke(query)[:k]

    except Exception as e:
        print(f"Self-Query failed for table '{target_table if 'target_table' in locals() else 'unknown'}', falling back to semantic search. Error: {e}")
        # Fallback: use appropriate retriever based on table
        if 'retriever' in locals():
            return retriever.invoke(query)[:k]
        else:
            # Create fallback retriever if error occurred during initialization
            # Re-initialize main retriever if needed, but self.retriever should exist
            return self.retriever.invoke(query)[:k]

    def _get_or_create_vectorstore(self, table_name: str):
        """
        Helper method to get or create a LanceDB vectorstore for any table.
        
        Args:
            table_name: Name of the table to create vectorstore for
            
        Returns:
            LanceDB vectorstore instance for the specified table
        """
        return LanceDB(
            connection=self.db,
            embedding=embeddings_model,
            table_name=table_name
        )


    def ingest_directory(self, path: str, recursive: bool = True, table_name: Optional[str] = None, progress_callback: Optional[Callable] = None) -> Tuple[bool, str, Optional[str]]:
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
            
            # Explicit callback after each file in directory ingestion
            if progress_callback:
                progress_callback()
                
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

    def _get_file_owner(self, path: str) -> str:
        """
        Gets the username of the file owner.
        """
        try:
            return pwd.getpwuid(os.stat(path).st_uid).pw_name
        except Exception:
            return "unknown"

    def _ingest_parent(self, file_path: str, table_name: str = None):
        try:
            target_table = table_name if table_name else self.vector_table_name
            print(f"--- 📥 INGESTING (Parent-Child): {file_path} into '{target_table}' ---")
            
            
            # NOTE: Schema migration disabled for Parent-Child strategy.
            # LangChain stores metadata in a 'metadata' struct, not as top-level columns.
            # Attempting to add top-level columns conflicts with the struct approach.
            # The metadata struct can hold any fields without schema conflicts.
            
            # 1. Load Docs
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                # Enable autodetect_encoding to handle various text encodings
                loader = TextLoader(file_path, autodetect_encoding=True)
            docs = loader.load()
            
            print(f"DEBUG: Document metadata keys: {list(docs[0].metadata.keys()) if docs else 'No docs'}")
            
            # 3. Get FRESH DB connection after schema changes (critical to avoid cached schema)
            fresh_db = get_db_connection()
            
            # 4. Init/Refresh Retriever with fresh connection
            vstore = LanceDB(
                connection=fresh_db,
                embedding=embeddings_model,
                table_name=target_table,
                mode="append"
            )
            
            target_retriever = ParentDocumentRetriever(
                vectorstore=vstore,
                docstore=self.docstore,
                child_splitter=self.child_splitter,
                parent_splitter=self.parent_splitter
            )
            
            # 4. Process Metadata & Add
            import json

            # CORE FIELDS are those we want explicit columns for (plus vector/text/id/source)
            # 'source' is inserted manually later.
            CORE_FIELDS = {'title', 'author', 'year', 'page', 'id', 'source', 'creationdate', 'creator', 'producer', 'moddate'}
            
            KEEP_AS_COLUMNS = {'author', 'page', 'source', 'title'} 

            for doc in docs:
                # 1. Author Fallback
                if 'author' not in doc.metadata or not doc.metadata['author']:
                    doc.metadata['author'] = self._get_file_owner(file_path)

                # 2. Title Fallback
                if 'title' not in doc.metadata or not doc.metadata['title']:
                     doc.metadata['title'] = os.path.basename(file_path)
                     
                # 3. Metadata Packing
                extra_metadata = {}
                keys_to_remove = []
                
                # Check for existing keys
                for key, value in list(doc.metadata.items()):
                    # Basic sanitization for all values
                    if value is None:
                        keys_to_remove.append(key)
                        continue
                        
                    # Clean strings
                    if not isinstance(value, (str, int, float, bool)):
                        str_val = str(value)
                        doc.metadata[key] = str_val
                        value = str_val

                    if key not in KEEP_AS_COLUMNS:
                        extra_metadata[key] = value
                        keys_to_remove.append(key)
                
                # Apply removal
                for key in keys_to_remove:
                    doc.metadata.pop(key, None)
                
                # Add the packed extras
                doc.metadata['extra_metadata'] = json.dumps(extra_metadata)
                
                # Ensure source is absolute path (Critical)
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

def ingest_path(path: str, strategy: Literal["naive", "parent"] = "naive", progress_callback: Optional[Callable] = None):
    """
    Ingests all files from a directory or a single file. (Naive strategy default)
    """
    print(f'ingesting path {path} with strategy {strategy}')
    
    path = os.path.abspath(os.path.expanduser(path))
    ingestor = NexusIngestor(strategy=strategy)
    
    if os.path.isfile(path):
        return ingestor.ingest_file(path, progress_callback=progress_callback)
    elif os.path.isdir(path):
        return ingestor.ingest_directory(path, progress_callback=progress_callback)
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