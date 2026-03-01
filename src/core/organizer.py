import os
import shutil
from typing import List, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.config import Config
from src.core.user_settings import get_setting
from src.embeddings.multimodal_onnx import get_multimodal_embedder
from src.rag.ingestion import ingest_file
from src.rag.ingestion_multimodal import purge_multimodal_rows
from src.core.database import WatchedPathsRepository


ALLOWED_CATEGORIES = [
    "Finance",
    "LegalContracts",
    "IdentityPersonal",
    "WorkDocuments",
    "Correspondence",
    "Programming",
    "DataLogs",
    "Technology Documents",
    "KeySecrets",
    "InstallersSoftware",
    "Images",
    "AudioVideo",
    "CreativeAssets",
    "BooksLibrary",
    "Travel",
    "Education",
    "Archives",
    "Unsorted",
]

CATEGORY_ALIASES = {
    "_Unsorted": "Unsorted",
    "Development": "Programming",
    "TechDocs": "Technology Documents",
}

IMAGE_CATEGORY_PROFILES = {
    "Finance": "an invoice, bill, receipt, bank statement, or financial document",
    "LegalContracts": "a contract, agreement, legal notice, or signed legal page",
    "IdentityPersonal": "an ID card, passport, certificate, or personal identity document",
    "WorkDocuments": "a generic office document, memo, or business paperwork",
    "Correspondence": "a letter, email screenshot, or message thread",
    "Programming": "source code, terminal output, an IDE window, or a developer tool screenshot",
    "DataLogs": "a log export, dashboard screenshot, chart, graph, or tabular system output",
    "Technology Documents": "technical documentation, a diagram, a manual, or a product/specification page",
    "KeySecrets": "passwords, keys, secret tokens, or sensitive credential material",
    "InstallersSoftware": "an installer wizard, software package screen, or app setup media",
    "Images": "a general photo, picture, or non-document visual image",
    "AudioVideo": "a media player frame, waveform, album art, or video still",
    "CreativeAssets": "a design mockup, illustration, poster, slide, or visual asset",
    "BooksLibrary": "a book cover, scanned book page, or publication page",
    "Travel": "a ticket, itinerary, hotel booking, map, or travel-related document",
    "Education": "lecture notes, study material, worksheet, slide, or classroom content",
    "Archives": "an old scanned document, archive page, or historical record",
    "Unsorted": "an ambiguous image that does not fit any specific category",
}


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
        self._clip_category_vectors = None

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
        category = self._categorize_file(filename, content_snippet, existing_folders, file_path=file_path)
        
        # 4. Move File
        dest_folder = os.path.join(watched_root, category)
        os.makedirs(dest_folder, exist_ok=True)
        
        dest_path = os.path.join(dest_folder, filename)
        dest_path = self._handle_collisions(dest_path)
        
        try:
            shutil.move(file_path, dest_path)
            print(f"Moved '{filename}' to '{category}'")
            
            # 5. Sync the shared multimodal index after the move.
            try:
                purge_multimodal_rows(file_path)
            except Exception as e:
                print(f"Failed to purge old multimodal record: {e}")

            try:
                ingest_file(dest_path, strategy="multimodal")
                print(f"Ingested '{dest_path}' into the shared multimodal index")
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

    def _categorize_file(self, filename: str, content: str, existing_folders: str, file_path: Optional[str] = None) -> str:
        # Fallback for empty/garbage content
        if not content or len(content) < 10:
            content = "[Content unreadable. RELY ON FILENAME ALONE]"

        categories = list(ALLOWED_CATEGORIES)
        category_list_str = ", ".join(categories)
        clip_hint = self._categorize_image_with_clip(file_path)

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
            Image Semantic Hint:
            {clip_hint}
            
            RULES:
            1. You MUST choose from the Allowed Categories list. Do not invent new ones.
            2. CONSIDER the file content AND filename (if it is meaningful) to decide the category.
            3. If the file is source code (py, js, sh, or other common code extension), choose 'Programming'.
            4. If the file is a log or data dump (json, csv, txt, log, etc.), choose 'DataLogs'.
            5. If the file is an invoice, receipt, or bill, choose 'Finance'.
            6. If the file is an image and you can't determine the category by name, choose 'Images'.
            7. If the file is an academic paper, technical manual, or documentation, choose 'Technology Documents'.
            8. If the file is txt, pdf, docx, epub, mobi, etc. and it is a book, choose 'BooksLibrary'
            9. If the file is user's personal document like aadhar card, pan card, passport, etc., choose 'IdentityPersonal'.
            10. If uncertain, unclear, or garbage, choose 'Unsorted'.
            11. For images, use the Image Semantic Hint when it is present, together with the filename.
            12. Output ONLY the category name.
            
            YOUR CHOICE:
            """
        )

        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "category_list": category_list_str,
                "filename": filename,
                "content": content,
                "clip_hint": clip_hint or "None",
            })
            
            # Simple cleanup to ensure it picked a valid one
            cleaned = response.strip().replace("'", "").replace('"', "")
            normalized = self._normalize_category(cleaned)
            if normalized:
                return normalized
            return "Unsorted"
            
        except Exception as e:
            print(f"LLM Categorization failed: {e}")
            return clip_hint or "Unsorted"

    def _categorize_image_with_clip(self, file_path: Optional[str]) -> Optional[str]:
        if not file_path:
            return None

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
            return None

        clip_embedder = get_multimodal_embedder()
        if not clip_embedder:
            return None

        try:
            from PIL import Image
        except ImportError:
            return None

        try:
            with Image.open(file_path) as image:
                image_vector = clip_embedder.embed_image(image.convert("RGB"))
        except Exception as exc:
            print(f"CLIP image categorization skipped for {file_path}: {exc}")
            return None

        category_vectors = self._get_clip_category_vectors(clip_embedder)
        if not category_vectors:
            return None

        best_category = None
        best_score = None
        for category, vector in category_vectors.items():
            score = sum(float(a) * float(b) for a, b in zip(image_vector, vector))
            if best_score is None or score > best_score:
                best_score = score
                best_category = category

        return best_category

    def _get_clip_category_vectors(self, clip_embedder):
        if self._clip_category_vectors is not None:
            return self._clip_category_vectors

        categories = list(IMAGE_CATEGORY_PROFILES.keys())
        prompts = [
            f"This image is {IMAGE_CATEGORY_PROFILES[category]}."
            for category in categories
        ]

        try:
            vectors = clip_embedder.embed_texts(prompts)
        except Exception as exc:
            print(f"CLIP text prompts unavailable for organizer: {exc}")
            self._clip_category_vectors = {}
            return self._clip_category_vectors

        self._clip_category_vectors = {
            category: vector
            for category, vector in zip(categories, vectors)
        }
        return self._clip_category_vectors

    def _normalize_category(self, value: str) -> Optional[str]:
        cleaned = (value or "").strip()
        if not cleaned:
            return None

        direct_matches = {}
        for category in ALLOWED_CATEGORIES:
            direct_matches[category.lower()] = category
        for alias, canonical in CATEGORY_ALIASES.items():
            direct_matches[alias.lower()] = canonical

        exact = direct_matches.get(cleaned.lower())
        if exact:
            return exact

        for needle, canonical in direct_matches.items():
            if needle in cleaned.lower():
                return canonical

        return None

    def _parse_category_from_response(self, response: str) -> str:
        if not response:
            return "Unsorted"

        import re
        
        # 1. Clean up "Chatty" prefixes like "Category: Invoices" or "I choose Invoices"
        # We look for the last capitalized word in the string, as categories are usually CamelCase.
        # But first, let's remove common distinct noise.
        clean_response = response.replace("Category:", "").replace("Answer:", "").strip()
        
        # 2. Extract the first valid CamelCase-ish word
        # Regex explanation: Look for a word starting with a letter, containing letters/numbers/underscores
        matches = re.findall(r'[a-zA-Z0-9_]+', clean_response)
        
        if not matches:
            return "Unsorted"
            
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
        return self._normalize_category(candidate) or "Unsorted"

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
