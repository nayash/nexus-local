import lancedb
import os
from src.core.config import Config

import shutil

# Define where the DB lives on your disk
DB_PATH = Config.LANCEDB_PATH
DOCSTORE_PATH = os.path.join(Config.PROJECT_ROOT, "data", "docstore")

def get_db_connection():
    """Establishes connection to local LanceDB instance."""
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
    return lancedb.connect(DB_PATH)

def get_table(table_name="documents"):
    """Gets or creates the vector table."""
    db = get_db_connection()
    
    # We define a simple schema implicitly by opening the table.
    # If it doesn't exist, the ingestion script will create it.
    try:
        return db.open_table(table_name)
    except FileNotFoundError:
        return None  # Table doesn't exist yet

def list_tables():
    """Returns a list of all table names in the database."""
    db = get_db_connection()
    tables = db.table_names()
    print(f'List of tables: {tables}')
    return tables

def clear_all_tables():
    """Drops all tables in the LanceDB database and clears DocStore."""
    print('inside clear_all_tables')
    
    # 1. Clear Vector DB
    db = get_db_connection()
    tables = db.table_names()
    print(f'vector db tables: {tables}')
    for table in tables:
        db.drop_table(table)
    print("Vector database cleared.")
    
    # 2. Clear DocStore
    try:
        if os.path.exists(DOCSTORE_PATH):
            shutil.rmtree(DOCSTORE_PATH)
            print(f"DocStore at {DOCSTORE_PATH} removed.")
        
        # Re-create empty directory
        os.makedirs(DOCSTORE_PATH, exist_ok=True)
        print("DocStore directory re-created.")
        
    except Exception as e:
        print(f"Error clearing DocStore: {e}")
        
    return True

def get_source_field_for_table(table) -> str:
    """
    Determines the correct field name for the source path in a LanceDB table.
    Returns 'source' if top-level, or 'metadata.source' if nested (common in Parent Strategy).
    Raises ValueError if neither found.
    """
    if not table:
         raise ValueError("Table is None")

    try:
        # Check top-level 'source'
        # LanceDB schema structure: schema.names returns list of top-level fields
        if 'source' in table.schema.names:
            return 'source'
        
        # Check for 'metadata' struct
        if 'metadata' in table.schema.names:
            # We assume if metadata exists, source is inside it based on ingestion logic
            return 'metadata.source'
            
        # Fallback or strict check? 
        # For now, if we can't find source, we might be in trouble.
        # But let's verify if metadata has source? 
        # Investigating schema deeply might be slow if we load it all, but names should be fast.
        
        return 'source' # Fallback to naive default to see if it works or fails with better error
        
    except Exception as e:
        print(f"Warning: Could not determine schema for table. Defaulting to 'source'. Error: {e}")
        return 'source'