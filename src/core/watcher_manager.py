import os
from typing import Tuple
from src.core.database import WatchedPathsRepository
from src.core.services.watcher import WatcherService
from src.rag.ingestion import ingest_path
from src.rag.ingestion_multimodal import purge_multimodal_prefix

class WatcherManager:
    """
    High-level manager for the file watcher system.
    Handles the 'Purge & Takeover' logic and coordinates between DB, Ingestion, and WatcherService.
    """
    def __init__(self):
        self.repo = WatchedPathsRepository()
        self.service = WatcherService()

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

        # 2. PURGE: Remove existing multimodal records for this path prefix.
        print(f"--- 🧹 PURGING multimodal records for: {path} ---")
        try:
            purge_multimodal_prefix(path)
        except Exception as e:
            print(f"Warning: Purge step encountered error (non-fatal): {e}")

        # 3. RECORD: Keep a stable scope name in SQLite for future workspace support.
        safe_name = "watch:" + os.path.basename(path).strip().replace(" ", "_").lower()

        # 4. RECORD: Save to DB (Must be before organization for repo lookups)
        try:
            self.repo.add_watched_path(
                path,
                safe_name,
                strategy="organize_and_ingest",
                watch_mode="organize_and_ingest",
                recursive=False,
            )
        except Exception as e:
            return False, f"Database save failed: {str(e)}"
        
        # 5. INITIAL ORGANIZE & INGEST
        print(f"--- 📥 INITIALIZING WATCHED FOLDER: {path} -> {safe_name} ---")
        try:
            # Semantically organize existing top-level files
            self._organize_existing_files(path, safe_name)
            
            # Then perform full ingestion for the remaining structure (subfolders etc)
            ingest_path(path, strategy="multimodal")
            
        except Exception as e:
            # Rollback DB if possible? For now just log and continue
            print(f"Initialization organization/ingestion error: {str(e)}")

        # 6. START: Start watching
        self.service.start()
        self.service.watch_path(path, watch_mode="organize_and_ingest", recursive=False)
        
        return True, f"Successfully started watching '{os.path.basename(path)}'"

    def _organize_existing_files(self, root_path: str, table_name: str):
        """
        Scans the root of a newly watched folder and organizes top-level files.
        """
        print(f"--- 🧠 INITIAL SCAN: Organizing existing files in {root_path} ---")
        files = [f for f in os.listdir(root_path) if os.path.isfile(os.path.join(root_path, f))]
        files = [f for f in files if not f.startswith(".")]
        print(f'files: {len(files)}')
        for filename in files:
            file_path = os.path.join(root_path, filename)
            try:
                # Synchronous organization during startup
                self.service.organizer.organize_file(file_path, root_path, table_name=table_name)
            except Exception as e:
                print(f"Error organizing existing file {filename}: {e}")

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
