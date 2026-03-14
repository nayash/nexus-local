import os
import time
import threading
from typing import Dict, Set, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.core.organizer import Organizer
from src.core.database import WatchedPathsRepository
from src.rag.ingestion import ingest_file


_WORKSPACE_INGEST_STRATEGIES = {"workspace_ingest", "ingest_only"}

class NexusFileEventHandler(FileSystemEventHandler):
    """
    Handles file system events for watched directories.
    """
    def __init__(
        self,
        root_path: str,
        organizer: Optional[Organizer],
        ignore_list: Set[str],
        watch_mode: str = "organize_and_ingest",
        workspace_id: Optional[str] = None,
    ):
        self.root_path = root_path
        self.organizer = organizer
        self.ignore_list = ignore_list
        self.watch_mode = (watch_mode or "organize_and_ingest").strip().lower()
        self.workspace_id = (workspace_id or "").strip() or None
        self.last_events: Dict[str, float] = {}
        self.debounce_seconds = 2.0

    def on_created(self, event):
        if event.is_directory:
            return
        self._process_event(event.src_path, "created")

    def on_modified(self, event):
        if event.is_directory:
            return
        self._process_event(event.src_path, "modified")

    def _process_event(self, file_path: str, event_type: str):
        # 1. Check Ignore List
        if file_path in self.ignore_list:
            if event_type == "modified":
                # Remove from ignore list after modification if needed, 
                # but better to keep ignoring if we are the ones touching it.
                # Just return for now.
                return

        # 2. Debounce
        current_time = time.time()
        last_time = self.last_events.get(file_path, 0)
        
        if current_time - last_time < self.debounce_seconds:
            return
        
        self.last_events[file_path] = current_time
        
        # 3. Validation
        if os.path.basename(file_path).startswith('.'):
            return
            
        # Wait for file write to complete
        time.sleep(1.0)
        
        try:
            if not os.path.exists(file_path):
                return
            if os.path.getsize(file_path) == 0:
                print(f"Skipping empty file: {file_path}")
                return 
        except OSError:
            return

        # 4. Trigger Organizer in a separate thread
        print(f"Detected change in '{file_path}'. Processing...")

        if self.watch_mode in _WORKSPACE_INGEST_STRATEGIES:
            threading.Thread(
                target=ingest_file,
                kwargs={"file_path": file_path, "workspace_id": self.workspace_id or "global"},
                daemon=True,
            ).start()
            return

        if self.organizer is None:
            return

        threading.Thread(
            target=self.organizer.organize_file,
            args=(file_path, self.root_path),
            daemon=True,
        ).start()


class WatcherService:
    """
    Singleton service to manage file watchers.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WatcherService, cls).__new__(cls)
            cls._instance.observer = Observer()
            cls._instance.watches = {} # path -> watch
            cls._instance.organizer = None
            cls._instance.ignore_list = set()
            cls._instance.repo = WatchedPathsRepository()
            cls._instance.handlers = {} # path -> handler
            cls._instance._loaded_from_db = False
        return cls._instance

    def _get_organizer(self) -> Organizer:
        if self.organizer is None:
            self.organizer = Organizer()
        return self.organizer

    def start(self):
        """Starts the observer."""
        if not self.observer.is_alive():
            try:
                 self.observer.start()
                 print("WatcherService started.")
                 should_autoload = os.getenv("NEXUS_WATCHER_AUTOLOAD", "true").strip().lower() in {"1", "true", "yes", "on"}
                 if should_autoload and not os.getenv("PYTEST_CURRENT_TEST") and not self._loaded_from_db:
                     self._load_from_db()
                     self._loaded_from_db = True
            except RuntimeError:
                pass

    def stop(self):
        """Stops the observer."""
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join(timeout=5)
            if self.observer.is_alive():
                print("WatcherService stop timed out; observer still alive.")
            else:
                print("WatcherService stopped.")

    def watch_path(
        self,
        path: str,
        *,
        watch_mode: str = "organize_and_ingest",
        workspace_id: Optional[str] = None,
        recursive: bool = False,
    ):
        """Adds a path to be watched."""
        if path in self.watches:
            print(f"Already watching: {path}")
            return

        if not os.path.exists(path):
            print(f"Path not found: {path}")
            return

        # Create a handler specifically for this path to pass the root context
        normalized_mode = (watch_mode or "organize_and_ingest").strip().lower()
        organizer = None if normalized_mode in _WORKSPACE_INGEST_STRATEGIES else self._get_organizer()
        handler = NexusFileEventHandler(
            path,
            organizer,
            self.ignore_list,
            watch_mode=normalized_mode,
            workspace_id=workspace_id,
        )

        watch = self.observer.schedule(handler, path, recursive=bool(recursive))
        self.watches[path] = watch
        self.handlers[path] = handler
        print(f"Started watching: {path}")

    def unwatch_path(self, path: str):
        """Removes a path from monitoring."""
        if path in self.watches:
            self.observer.unschedule(self.watches[path])
            del self.watches[path]
            del self.handlers[path]
            print(f"Stopped watching: {path}")

    def _load_from_db(self):
        """Loads active watched paths from DB."""
        paths = self.repo.get_watched_paths()
        for p in paths:
            if p['status'] == 'active':
                self.watch_path(
                    p['path'],
                    watch_mode=p.get("watch_mode") or p.get("strategy") or "organize_and_ingest",
                    workspace_id=p.get("workspace_id"),
                    recursive=bool(p.get("recursive")),
                )

    def add_to_ignore(self, path: str):
        self.ignore_list.add(path)
