You are an expert Python developer specialized in Flet, AsyncIO, and Local RAG systems. You are tasked with implementing the "Nexus Organizer & Watcher" module.

Objective
Create a system that allows users to "Watch" specific local folders. When a folder is watched:

Ingest: Nexus takes ownership of the folder, indexing all its contents into a dedicated vector table.

Monitor: A background service watches for file changes (creation/modification).

Organize: New files are automatically analyzed by the LLM and moved into context-appropriate subfolders (e.g., /Invoices, /Scripts) or a _Unsorted folder if ambiguous.

Sync: The Vector Database (LanceDB) updates automatically to reflect these moves/deletes.

Implementation Plan
Phase 1: Database Schema (Persistence)
Action: Modify src/core/database.py.

Task: Create a new SQLite table watched_paths to persist state across restarts.

Schema:

id (TEXT PRIMARY KEY)

path (TEXT): Absolute path to the watched root.

table_name (TEXT): The dedicated LanceDB table name (e.g., watched_home_user_docs).

status (TEXT): 'active' or 'paused'.

last_scan (REAL): Timestamp.

strategy (TEXT): Default to 'organize_and_ingest'.

Phase 2: UI - Settings View Update
Action: Update src/ui/views/settings_view.py.

Task: Add a "Watched Folders" section.

List: Show currently watched paths with a "Remove/Stop" button.

Add: A button that opens FilePicker (directory mode).

Logic: When a folder is selected:

Check if it's already watched.

Trigger the "Purge & Takeover" background task (detailed below).

Add entry to SQLite.

Start the Watcher Service.

Phase 3: The "Purge & Takeover" Logic (Critical)
Context: Avoid duplicate search results (one from old manual ingest, one from new watcher).

Task: Implement initialize_watcher(path) in src/core/watcher_manager.py.

Step A (Purge): Search the generic documents & other folder specific table in LanceDB. Delete ANY record where source starts with the target path.

Step B (Create): Generate a unique table name (e.g., watched_simplified_folder_path_{hash}).

Step C (Ingest): Call the existing ingest_path function to index the folder into this new table.

Step D (Record): Save metadata to watched_paths SQLite table.

Phase 4: The Watcher Service (Background Daemon)
Action: Create src/core/services/watcher.py.

Library: Use watchdog.observers.

Logic:

Run as a singleton service managed by main.py.

On App Start: Load 'active' paths from SQLite and start observers.

Event Handling: Listen for FileCreated and FileModified.

Debounce: Wait 2-3 seconds after an event to ensure file write is complete.

Empty Check: If os.path.getsize(f) == 0, IGNORE it. (Wait for the subsequent 'modified' event when content is written).

Loop Prevention: Maintain an in-memory IGNORE_LIST. If Nexus moves a file itself, add destination to list so the watcher ignores the resulting event.

Phase 5: The Organizer Logic (LLM Agent)
Action: Create src/core/organizer.py.

Task: When a valid file event occurs:

Read: Extract first 2KB of text.

Context: List existing subfolders in the root watched_path.

Prompt (Llama 3.1):

"Analyze this file content. Existing categories are: {current_folders}. Return the exact name of the best fitting category. If none fit, create a short, specific name. If unclear/garbage, return '_Unsorted'."

Move:

Create the subfolder if missing.

Safety: If dest/file.txt exists, rename to file_1.txt.

Move the file.

DB Sync:

Delete the record for the old path from the dedicated LanceDB table.

Ingest the new path into the dedicated LanceDB table.

Technical Constraints & Rules
Reuse Existing Tools:

Use src.rag.ingestion NexusIngestor class for indexing.

Use src.rag.storage.get_table for DB access.

Use src.core.database for SQLite interactions.

And any other existing code.

Concurrency Safety:

The Watcher runs in a threading.Thread or multiprocessing.Process.

Database writes (SQLite & LanceDB) must be thread-safe or locked.

UI updates (Notifications) must use asyncio.create_task or be dispatched to the main thread.

Flat Hierarchy:

The Organizer should only create folders at the Root Level of the watched directory (Depth=1). Do not create deep nested structures (e.g., /Work/2024/Project). Keep it flat: /Work_Projects.

Fail-Safe:

If the LLM fails or crashes, move file to _Unsorted.

Never delete a file from the disk unless explicitly instructed by user (which is not part of this feature).

Notifications:

Use the existing NotificationManager to alert the user: "Moved 'invoice.pdf' to 'Finance'".

Output Deliverables
Provide the code for:

src/core/database.py (Schema update)

src/core/services/watcher.py (The logic)

src/core/organizer.py (The logic)

src/ui/views/settings_view.py (The integration)