import os
import time
import shutil
import unittest
import threading
from unittest.mock import MagicMock, patch

# Ensure we can import from src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.database import WatchedPathsRepository, init_db, DB_PATH
from src.core.watcher_manager import WatcherManager
from src.core.services.watcher import WatcherService
from src.core.organizer import Organizer

class TestWatcherHeadless(unittest.TestCase):
    def setUp(self):
        # Let's create a temp test directory
        self.test_dir = os.path.abspath("test_watch_folder")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        # Initialize components
        self.manager = WatcherManager()
        
    def tearDown(self):
        # Stop everything
        self.manager.service.stop()
        
        # Clean up test dir
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
        # Clean up DB entries for this test path
        repo = WatchedPathsRepository()
        paths = repo.get_watched_paths()
        for p in paths:
            if p['path'] == self.test_dir:
                repo.remove_watched_path(p['id'])

    @patch('src.core.organizer.ChatOllama')
    @patch('src.core.watcher_manager.ingest_path') # Mock ingestion
    def test_watcher_lifecycle(self, mock_ingest_path, mock_llm_cls):
        print("\n--- Testing Watcher Implementation ---")
        mock_ingest_path.return_value = (True, "ok", None)
        
        watcher_service = WatcherService()
        
        # START WATCHING
        print("Initializing watch path...")
        success, msg = self.manager.initialize_path(self.test_dir)
        self.assertTrue(success, f"Failed to init path: {msg}")
        print(f"Watch initialized: {msg}")
        
        # Verify DB entry
        repo = WatchedPathsRepository()
        paths = repo.get_watched_paths()
        found = False
        for p in paths:
            if p['path'] == self.test_dir:
                found = True
                break
        self.assertTrue(found, "Path not found in DB")
        
        # Verify Watcher Service has it
        self.assertIn(self.test_dir, watcher_service.watches)
        print("Service is watching the path.")

        # SIMULATE FILE CREATION
        # Mock the organizer.organize_file method
        original_organize = watcher_service.organizer.organize_file
        watcher_service.organizer.organize_file = MagicMock()
        
        test_file = os.path.join(self.test_dir, "test_doc.txt")
        with open(test_file, "w") as f:
            f.write("This is a test document content snippet.")
            
        print("Created test file. Waiting for event debounce...")
        
        # WatcherService uses 2.0s debounce + 1.0s sleep in handler.
        time.sleep(4)
        
        if watcher_service.organizer.organize_file.called:
            print("SUCCESS: organize_file was called!")
            args, _ = watcher_service.organizer.organize_file.call_args
            self.assertEqual(args[0], test_file)
            self.assertEqual(args[1], self.test_dir)
        else:
             print("FAILURE: organize_file was NOT called.")
             
        # Restore original method
        watcher_service.organizer.organize_file = original_organize
        
        # STOP WATCHING
        path_id = [p['id'] for p in repo.get_watched_paths() if p['path'] == self.test_dir][0]
        success, msg = self.manager.stop_watching(path_id)
        self.assertTrue(success)
        print("Stopped watching.")
        
        self.assertNotIn(self.test_dir, watcher_service.watches)

if __name__ == '__main__':
    unittest.main()
