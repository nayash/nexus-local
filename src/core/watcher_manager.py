import os
import shutil
import uuid
from typing import List, Dict, Tuple
from src.core.database import WatchedPathsRepository
from src.core.services.watcher import WatcherService
from src.rag.ingestion import ingest_path, NexusIngestor
from src.rag.storage import get_db_connection, get_table

class WatcherManager:
    """
    High-level manager for the file watcher system.
    Handles the 'Purge & Takeover' logic and coordinates between DB, Ingestion, and WatcherService.
    """
    def __init__(self):
        self.repo = WatchedPathsRepository()
        self.service = WatcherService()
        # Ensure service is started
        self.service.start()

    def initialize_path(self, path: str) -> Tuple[bool, str]:
        """
        Sets up a new watched path with 'Purge & Takeover' logic.
        """
        path = os.path.abspath(path)
        if not os.path.exists(path):
            return False, "Path does not exist."

        # 1. Check if already watched
        existing = self.repo.get_watched_paths()
        for p in existing:
            if p['path'] == path:
                 # If paused, just resume? 
                 # For now, let's say "Already watched".
                 return False, "Path is already being watched."

        # 2. PURGE: Remove legacy records from generic tables
        # We need to find where this path might have been ingested before.
        # Usually 'documents' is the default.
        # Also check other tables? 
        # Ideally we only purge from 'documents' for now as that's the naive default.
        
        print(f"--- 🧹 PURGING legacy records for: {path} ---")
        try:
            # We assume the default table is 'documents' and maybe 'nexus_parent_child'
            # We should clear from both to be safe? 
            # Or just 'documents' if that was the old default.
            # Let's try to purge from 'documents' (naive) and 'nexus_parent_child' (parent)
            
            tables_to_check = ["documents", "nexus_parent_child"]
            for tbl_name in tables_to_check:
                tbl = get_table(tbl_name)
                if tbl:
                    # LanceDB delete syntax: "source LIKE 'path%'" matches directory
                    # Note: source is absolute path.
                    # We want to delete everything starting with this path.
                    # SQL style: source LIKE '/abs/path/%'
                    escape_path = path.replace("'", "''") 
                    # Add trailing slash to avoid partial matches (e.g. /foo matching /foobar)
                    # But also match the folder itself if it was ingested as a file?? (Not possible for strict dir)
                    # For a directory ingest, all files inside have source starting with path.
                    
                    # Construct filter
                    # LanceDB SQL filter support is limited, but string matching works.
                    # "source LIKE '...%'"
                    
                    filter_query = f"source LIKE '{escape_path}%'"
                    tbl.delete(filter_query)
                    print(f"Purged records from '{tbl_name}' matching '{path}'")

        except Exception as e:
            print(f"Warning: Purge step encountered error (non-fatal): {e}")

        # 3. CREATE: Generate dedicated table name
        # We use the sanitizer from Ingestor or just make one here.
        safe_name = "watched_" + os.path.basename(path).strip().replace(" ", "_").lower()
        # Append hash to ensure uniqueness
        safe_name = f"{safe_name}_{uuid.uuid4().hex[:8]}"
        
        # 4. INGEST: Initial full ingestion
        print(f"--- 📥 INITIALIZING WATCHED FOLDER: {path} -> {safe_name} ---")
        try:
            # We use 'parent' strategy by default for watched folders as it's better.
            ingestor = NexusIngestor(strategy="parent")
            
            # Ingest directory with custom table name
            ingestor.ingest_directory(path, recursive=True, table_name=safe_name)
            
        except Exception as e:
            return False, f"Ingestion failed: {str(e)}"

        # 5. RECORD: Save to DB
        try:
            self.repo.add_watched_path(path, safe_name, strategy="organize_and_ingest")
        except Exception as e:
            return False, f"Database save failed: {str(e)}"

        # 6. START: Start watching
        self.service.watch_path(path)
        
        return True, f"Successfully started watching '{os.path.basename(path)}'"

    def stop_watching(self, path_id: str) -> Tuple[bool, str]:
        """
        Stops watching a path and removes it from list.
        Does NOT delete the data (user should do that manually if desired).
        """
        # Get path details
        paths = self.repo.get_watched_paths()
        target = None
        for p in paths:
            if p['id'] == path_id:
                target = p
                break
        
        if not target:
            return False, "Watch ID not found."

        # Stop service
        self.service.unwatch_path(target['path'])
        
        # Remove from SQLite
        self.repo.remove_watched_path(path_id)
        
        return True, f"Stopped watching '{target['path']}'"

    def get_watched_paths(self):
        return self.repo.get_watched_paths()
