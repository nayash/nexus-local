
import os
from src.core.config import Config

def get_dir_size(path: str) -> int:
    """
    Calculates the total size of a directory in bytes.
    """
    total_size = 0
    try:
        if not os.path.exists(path):
            return 0
            
        if os.path.isfile(path):
            return os.path.getsize(path)

        for entry in os.scandir(path):
            if entry.is_file():
                total_size += entry.stat().st_size
            elif entry.is_dir():
                total_size += get_dir_size(entry.path)
    except Exception as e:
        print(f"Error calculating size for {path}: {e}")
        return 0
    return total_size

def format_size(size_bytes: int) -> str:
    """
    Formats bytes into a human-readable string (KB, MB, GB).
    """
    if size_bytes == 0:
        return "0 B"
        
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def get_total_storage_usage() -> str:
    """
    Calculates the total storage usage of indexed documents.
    Includes:
    - SQLite Database
    - Vector Database (LanceDB)
    - Document Store (Pickled Parent Docs)
    """
    
    # 1. SQLite DB
    sqlite_size = get_dir_size(Config.SQLITE_PATH)
    
    # 2. Vector DB
    lancedb_size = get_dir_size(Config.LANCEDB_PATH)
    
    # 3. DocStore
    docstore_size = get_dir_size(Config.DOCSTORE_PATH)
    
    total_size = sqlite_size + lancedb_size + docstore_size
    
    formatted_total = format_size(total_size)
    
    # Optional: We could return a breakdown if needed, but for now just total string
    return f"Total Storage: {formatted_total}"
