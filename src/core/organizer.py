import os
import shutil
import time
from typing import List, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.config import Config
from src.core.user_settings import get_setting
from src.rag.ingestion import ingest_file
from src.core.database import WatchedPathsRepository, ChatRepository # Just to be safe, though not needed directly here yet
from src.rag.storage import get_db_connection, get_table, get_source_field_for_table


class Organizer:
    """
    Organizes files into subfolders using an LLM to determine the best category.
    """
    def __init__(self):
        self.llm = ChatOllama(
            model=get_setting("model_name", "llama3.1"),
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
                        source_field = get_source_field_for_table(tbl)
                        # Escape single quotes for filter
                        escaped_old_path = file_path.replace("'", "''")
                        # Count records before deletion to report accurately
                        filter_expr = f"{source_field} = '{escaped_old_path}'"
                        results = tbl.search().where(filter_expr).limit(None).to_list()
                        deleted_count = len(results)

                        tbl.delete(filter_expr)

                        print(f"Deleted {deleted_count} record(s) for '{file_path}' from '{table_name}' using field '{source_field}'")
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
        """
        Smartly reads a text snippet from a file. 
        Handles PDFs using pypdf and fails gracefully for other binaries.
        """
        import os
        from pypdf import PdfReader

        ext = os.path.splitext(file_path)[1].lower()

        try:
            # 1. Handle PDF specifically
            if ext == '.pdf':
                try:
                    reader = PdfReader(file_path)
                    text = ""
                    # Read first 2 pages max to get context
                    for page in reader.pages[:2]: 
                        text += page.extract_text() or ""
                        if len(text) > max_chars:
                            break
                    return text[:max_chars].strip()
                except Exception as e:
                    print(f"Error reading PDF {file_path}: {e}")
                    return "[Unreadable PDF Content]"

            # 2. Handle known binary extensions to avoid "garbage" characters
            # You can expand this list (images, zips, executables)
            binary_exts = {'.png', '.jpg', '.jpeg', '.gif', '.zip', '.exe', '.bin', '.pyc'}
            if ext in binary_exts:
                return "[Binary File - Categorize based on Filename]"

            # 3. Default Text Read (with utf-8 replacement to avoid crashes)
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(max_chars)
                # Check if it looks like binary garbage (too many null bytes)
                if '\0' in content:
                    return "[Binary Content]"
                return content.strip()
                
        except Exception as e:
            print(f"Failed to read snippet for {file_path}: {e}")
            return ""

    def _get_existing_folders(self, root: str) -> str:
        folders = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith('.')]
        if not folders:
            return "None"
        return ", ".join(folders)

    def _categorize_file(self, filename: str, content: str, existing_folders: str) -> str:
        # Fallback for empty/garbage content
        if not content or len(content) < 10:
            content = "[Content unreadable. RELY ON FILENAME ALONE]"

        # The Fixed Taxonomy
        categories = [
            "Finance", "LegalContracts", "IdentityPersonal", "WorkDocuments", "Correspondence",
            "Programming", "DataLogs", "Technology Documents", "KeySecrets", "InstallersSoftware",
            "Images", "AudioVideo", "CreativeAssets",
            "BooksLibrary", "Travel", "Education", "Archives", "Unsorted"
        ]
        
        category_list_str = ", ".join(categories)

        prompt = ChatPromptTemplate.from_template(
            """
            You are a file sorting engine.
            
            Task: Assign the file to ONE of the Allowed Categories below.
            
            ALLOWED CATEGORIES:
            [{category_list}]
            
            FILE CONTEXT:
            Filename: {filename}
            Content Snippet:
            {content}
            
            RULES:
            1. You MUST choose from the Allowed Categories list. Do not invent new ones.
            2. CONSIDER the file content AND filename (if it is meaningful) to decide the category.
            3. If the file is source code (py, js, sh, or other common code extension), choose 'Development'.
            4. If the file is a log or data dump (json, csv, txt, log, etc.), choose 'DataLogs'.
            5. If the file is an invoice, receipt, or bill, choose 'Finance'.
            6. If the file is an image and you can't determine the category by name, choose 'Images'.
            7. If the file is an academic paper or manual (txt, pdf, doc, docx, etc.), choose 'TechDocs'.
            8. If the file is txt, pdf, docx, epub, mobi, etc. and it is a book, choose 'BooksLibrary'
            9. If the file is user's personal document like aadhar card, pan card, passport, etc., choose 'IdentityPersonal'.
            10. If uncertain, unclear, or garbage, choose 'Unsorted'.
            11. Output ONLY the category name.
            
            YOUR CHOICE:
            """
        )

        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "category_list": category_list_str,
                "filename": filename,
                "content": content
            })
            
            # Simple cleanup to ensure it picked a valid one
            cleaned = response.strip().replace("'", "").replace('"', "")
            
            # Validation: If LLM hallucinated, force Unsorted
            # (Fuzzy match could be added here, but exact match is safer for now)
            if cleaned not in categories:
                # Try to find a partial match (e.g., LLM said "Category: Finance")
                for valid_cat in categories:
                    if valid_cat in cleaned:
                        return valid_cat
                return "_Unsorted"
                
            return cleaned
            
        except Exception as e:
            print(f"LLM Categorization failed: {e}")
            return "_Unsorted"

    def _parse_category_from_response(self, response: str) -> str:
        if not response:
            return "_Unsorted"

        import re
        
        # 1. Clean up "Chatty" prefixes like "Category: Invoices" or "I choose Invoices"
        # We look for the last capitalized word in the string, as categories are usually CamelCase.
        # But first, let's remove common distinct noise.
        clean_response = response.replace("Category:", "").replace("Answer:", "").strip()
        
        # 2. Extract the first valid CamelCase-ish word
        # Regex explanation: Look for a word starting with a letter, containing letters/numbers/underscores
        matches = re.findall(r'[a-zA-Z0-9_]+', clean_response)
        
        if not matches:
            return "_Unsorted"
            
        # Usually the first word is the best bet if we stripped "Category:"
        candidate = matches[0]
        
        # 3. Final Sanity Check
        # If the candidate is just a common stop word (like "The", "A", "Based"), try the next one
        stop_words = {"The", "A", "An", "Based", "I", "This", "File", "Content"}
        if candidate in stop_words and len(matches) > 1:
            candidate = matches[1]
            
        # 4. Length Limit & Sanitize
        candidate = candidate[:30] # Categories shouldn't be massive
        
        # Ensure it starts with a letter or underscore
        if not candidate[0].isalpha() and candidate[0] != '_':
            candidate = "Docs_" + candidate

        print(f'Parsed category: {candidate}')
        return candidate

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
