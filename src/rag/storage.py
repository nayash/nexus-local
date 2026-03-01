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
