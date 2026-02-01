import os
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from src.core.config import Config
from src.rag.storage import get_db_connection, get_table

# Initialize the lightweight embedding model
embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=Config.OLLAMA_BASE_URL
)

def ingest_file(file_path: str, table_name: str = "documents"):
    """
    Reads a file, chunks it, embeds it, and saves to LanceDB.
    """
    try:
        print(f"--- 📥 INGESTING: {file_path} INTO TABLE: {table_name} ---")
        
        # 1. Load the file based on extension
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
            
        docs = loader.load()
        print(f"   -> Loaded {len(docs)} pages/documents.")

        # 2. Split into chunks (Small enough for LLM context window)
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
                # DEDUPLICATION LOGIC: Delete existing chunks for this file before adding new ones
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
            # Fallback for creation if get_table failed or other issues
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
        return False, str(e), None  # Return 3 values even on error

def ingest_path(path: str):
    """
    Ingests all files from a directory or a single file.
    - If path is a directory: Ingests into a table named after the directory (sanitized).
    - If path is a file: Ingests into 'documents' table.
    """
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
        # e.g., /home/user/docs -> folder_home_user_docs
        sanitized_name = "folder_" + path.strip(os.sep).replace(os.sep, "_").replace(".", "").replace("-", "_").replace(" ", "_").lower()
        # Ensure it's not too long and is valid
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
    
    for file_path in files_to_ingest:
        # Pass the determined table name
        success, chunks, _ = ingest_file(file_path, table_name=table_name)
        if success:
            successful_files += 1
            total_chunks += chunks
            
    # For Focus Mode, if we ingested exactly one file, return its absolute path
    final_path = os.path.abspath(files_to_ingest[0]) if len(files_to_ingest) == 1 else None
    
    msg = f"Successfully ingested {successful_files} files ({total_chunks} chunks) into '{table_name}'."
    return True, msg, final_path

# Quick test block
def init_knowledge():
    # with open("test_knowledge.txt", "w") as f:
    #     f.write("Project Nexus-Local is a private AI search engine designed by an Elite Engineer, Asutosh Nayak.")
    
    # Resolve path relative to this file to ensure it works when packaged
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up 2 levels: src/rag -> src -> project_root
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    data_file = os.path.join(project_root, "data", "nexus-identity.txt")
    
    ingest_file(data_file)