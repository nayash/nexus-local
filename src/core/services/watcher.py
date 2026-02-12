import os
import time
import threading
from typing import Dict, Set, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.core.organizer import Organizer
from src.core.database import WatchedPathsRepository

class NexusFileEventHandler(FileSystemEventHandler):
    """
    Handles file system events for watched directories.
    """
    def __init__(self, root_path: str, organizer: Organizer, ignore_list: Set[str]):
        self.root_path = root_path
        self.organizer = organizer
        self.ignore_list = ignore_list
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
        print(f"Detected change in '{file_path}'. Organizing...")
        
        # We need to ensure we don't trigger recursively.
        # The organizer writes to the same folder structure, so the destination path 
        # WILL trigger a new event.
        # We must add destination path to ignore_list BEFORE moving?
        # The Organizer should probably return the destination path so we can ignore it?
        # Or better: The Organizer moves it INTO a subfolder. If we watch recursive=False, 
        # and the organizer moves it DEEPER, then we are fine?
        # Constraint: "The Organizer should only create folders at the Root Level... Keep it flat"
        # Wait, if we watch root (recursive=False), and we move file to root/Subfolder, 
        # then the move operation is a DELETE from root (if moving) or CREATE in Subfolder.
        # If recursive=False, we only see events in root.
        # Moving from root/file.txt to root/Category/file.txt:
        # 1. MOVED_FROM root/file.txt (We see this)
        # 2. MOVED_TO root/Category/file.txt (We DON'T see this if recursive=False?)
        # Actually standard `mv` might trigger passed events differently.
        # If we use shutil.move, it might be copy+delete.
        
        # If we use recursive=False, we only watch the files in the root.
        # If a file is created in root, we process it.
        # Use recursive=False as per "Keep it flat" implies we organize the root's mess.
        
        # Start organization task
        threading.Thread(target=self.organizer.organize_file, args=(file_path, self.root_path)).start()


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
            cls._instance.organizer = Organizer()
            cls._instance.ignore_list = set()
            cls._instance.repo = WatchedPathsRepository()
            cls._instance.handlers = {} # path -> handler
        return cls._instance

    def start(self):
        """Starts the observer."""
        if not self.observer.is_alive():
            try:
                 self.observer.start()
                 print("WatcherService started.")
                 self._load_from_db()
            except RuntimeError:
                pass

    def stop(self):
        """Stops the observer."""
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
            print("WatcherService stopped.")

    def watch_path(self, path: str):
        """Adds a path to be watched."""
        if path in self.watches:
            print(f"Already watching: {path}")
            return

        if not os.path.exists(path):
            print(f"Path not found: {path}")
            return

        # Create a handler specifically for this path to pass the root context
        handler = NexusFileEventHandler(path, self.organizer, self.ignore_list)
        
        # Recursive=False to only organize the "Inbox" (root) and not mess with subfolders
        watch = self.observer.schedule(handler, path, recursive=False)
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
                self.watch_path(p['path'])

    def add_to_ignore(self, path: str):
        self.ignore_list.add(path)
