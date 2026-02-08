import os
import shutil
import time
from typing import List, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.config import Config
from src.rag.ingestion import ingest_file
from src.core.database import WatchedPathsRepository, ChatRepository # Just to be safe, though not needed directly here yet
from src.rag.storage import get_db_connection, get_table

class Organizer:
    """
    Organizes files into subfolders using an LLM to determine the best category.
    """
    def __init__(self):
        self.llm = ChatOllama(
            model="llama3.1",
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0
        )
        self.repo = WatchedPathsRepository()

    def organize_file(self, file_path: str, watched_root: str, table_name: Optional[str] = None):
        """
        Analyzes and moves a file to a semantic subfolder.
        """
        print(f'organizing file: {file_path}')
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

        filename = os.path.basename(file_path)
        
        # 1. Read Content (First 2KB)
        content_snippet = self._read_file_snippet(file_path)
        if not content_snippet:
            print(f"Skipping empty or unreadable file: {filename}")
            return

        # 2. Get Existing Categories
        existing_folders = self._get_existing_folders(watched_root)
        
        # 3. Ask LLM
        category = self._categorize_file(filename, content_snippet, existing_folders)
        
        # 4. Move File
        dest_folder = os.path.join(watched_root, category)
        os.makedirs(dest_folder, exist_ok=True)
        
        dest_path = os.path.join(dest_folder, filename)
        dest_path = self._handle_collisions(dest_path)
        
        try:
            shutil.move(file_path, dest_path)
            print(f"Moved '{filename}' to '{category}'")
            
            # 5. Ingest
            # We need to find which table this watched path belongs to.
            # Ideally we pass this info or look it up.
            # For now, let's look it up or rely on the caller to handle ingestion if it's a batch?
            # The requirements say: "Ingest: Nexus takes ownership... indexing... Database Sync: Delete old, Ingest new"
            
            # Find the table name for this watched root
            # efficient way: The watcher service knows the table name. 
            # But let's look it up from DB if we can, or just take it as arg?
            # Let's verify with the repo.
            
            if not table_name:
                watched_paths = self.repo.get_watched_paths()
                for wp in watched_paths:
                    if wp["path"] == watched_root:
                        table_name = wp["table_name"]
                        break
            
            if table_name:
                # 6. Delete old record from LanceDB (sync move)
                try:
                    tbl = get_table(table_name)
                    if tbl:
                        # Escape single quotes for filter
                        escaped_old_path = file_path.replace("'", "''")
                        tbl.delete(f"source = '{escaped_old_path}'")
                        print(f"Deleted old record for '{file_path}' from '{table_name}'")
                except Exception as e:
                    print(f"Failed to delete old record: {e}")

                # 7. Ingest into the dedicated table
                try:
                    ingest_file(dest_path, table_name, strategy="parent")
                    print(f"Ingested '{dest_path}' into '{table_name}'")
                except Exception as e:
                    print(f"Ingestion failed for organized file: {e}")

        except Exception as e:
            print(f"Failed to move file: {e}")

    def _read_file_snippet(self, file_path: str, max_chars: int = 2000) -> str:
        try:
            with open(file_path, 'r', errors='ignore') as f:
                return f.read(max_chars)
        except Exception:
            return ""

    def _get_existing_folders(self, root: str) -> str:
        folders = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith('.')]
        if not folders:
            return "None"
        return ", ".join(folders)

    def _categorize_file(self, filename: str, content: str, existing_folders: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            """
            You are an expert file librarian. Analyze the following file to determine the best semantic folder category.
            
            Filename: {filename}
            Content Snippet (First 2KB):
            {content}
            
            Existing Categories: {existing_folders}
            
            Rules:
            1. Categorize based on CONTENT TYPE & file names if it's meaningful (e.g., 'Invoices', 'ResearchPapers', 'Logs', 'Contracts' etc.), NOT file format (e.g., 'PDFs', 'TextFiles').
            2. BANNED CATEGORIES (Do NOT use these): 'Documents', 'Files', 'Data', 'General', 'Misc', 'Others', 'Documentation'.
            3. Use an existing category ONLY if it is a PERFECT semantic match. Otherwise, create a new specific one.
            4. Format: CamelCase (e.g., 'ResearchPapers', 'LegalDocs', 'ServerLogs').
            5. Return ONLY the category name. No markdown.
            
            Examples:
            - Content is an academic paper -> 'ResearchPapers'
            - Content is a receipt/bill -> 'Invoices'
            - Content is a python script -> 'Scripts'
            - Content is a server log -> 'Logs'
            - Content is an insurance policy -> 'InsuranceDocs'
            """
        )
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "filename": filename,
                "content": content,
                "existing_folders": existing_folders
            })
            return self._parse_category_from_response(response)
        except Exception as e:
            print(f"LLM Categorization failed: {e}")
            return "_Unsorted"

    def _parse_category_from_response(self, response: str) -> str:
        if not response:
            return "_Unsorted"

        # 1. Extract from **bold** if present (common LLM pattern)
        if "**" in response:
            import re
            match = re.search(r'\*\*(.*?)\*\*', response)
            if match:
                candidate = match.group(1).strip()
                if candidate:
                    response = candidate
        
        # 2. Extract first non-empty line if it's still multiline
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if not lines:
            return "_Unsorted"
        
        # Taking the first line as the category candidate
        raw_category = lines[0]

        # 3. Sanitize (Alphanumeric + Underscore/Hyphen only)
        # Replace common separators with underscore
        sanitized = raw_category.replace(" ", "_").replace("/", "_").replace("\\", "_")
        
        # Remove any other non-allowed characters
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '', sanitized)
        
        # 4. Length Limit (Linux max is 255, but we want short categories)
        max_length = 50
        cleaned = sanitized[:max_length]

        # 5. Final Fallback
        if not cleaned:
            return "_Unsorted"
            
        return cleaned

    def _handle_collisions(self, dest_path: str) -> str:
        """
        If file exists, append _1, _2, etc.
        """
        if not os.path.exists(dest_path):
            return dest_path
            
        base, ext = os.path.splitext(dest_path)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        return f"{base}_{counter}{ext}"
