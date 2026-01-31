import lancedb
import os
from src.core.config import Config

# Define where the DB lives on your disk
DB_PATH = Config.LANCEDB_PATH

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