import os
from typing import Callable, Literal, Optional

from src.core.config import Config
from src.rag.file_type_detector import detect_file_type
from src.rag.ingestion_multimodal import SUPPORTED_MULTIMODAL_EXTENSIONS, ingest_file_multimodal


def ingest_file(
    file_path: str,
    table_name: str = "",
    strategy: Literal["multimodal"] = "multimodal",
    workspace_id: str = "global",
):
    """
    Ingest a single file using the shared multimodal pipeline.

    `table_name` is accepted for compatibility with older call sites but ignored.
    """
    _ = table_name, strategy
    return ingest_file_multimodal(file_path, workspace_id=workspace_id)


def ingest_path(
    path: str,
    strategy: Literal["multimodal"] = "multimodal",
    progress_callback: Optional[Callable] = None,
    workspace_id: str = "global",
):
    """
    Ingest a file or directory using the shared multimodal pipeline.
    """
    _ = strategy
    print(f"ingesting path {path} with strategy multimodal")

    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        return ingest_file_multimodal(path, progress_callback=progress_callback, workspace_id=workspace_id)

    if os.path.isdir(path):
        files_to_ingest = []
        for root, _, files in os.walk(path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if file_name.lower().endswith(SUPPORTED_MULTIMODAL_EXTENSIONS):
                    files_to_ingest.append(file_path)
                    continue
                detected = detect_file_type(file_path)
                if detected.ingestible:
                    files_to_ingest.append(file_path)

        if not files_to_ingest:
            return False, "No supported files found.", None

        total_chunks = 0
        successful_files = 0
        for file_path in files_to_ingest:
            success, chunk_count, _ = ingest_file_multimodal(
                file_path,
                progress_callback=progress_callback,
                workspace_id=workspace_id,
            )
            if success:
                successful_files += 1
                total_chunks += chunk_count

        return True, f"Successfully ingested {successful_files} files ({total_chunks} chunks).", None

    return False, "Path does not exist.", None


def init_knowledge():
    """
    Seed the local knowledge base with the Nexus identity file when present.
    """
    candidates = [
        os.path.join(Config.DATA_DIR, "nexus-identity.txt"),
        os.path.join(Config.PROJECT_ROOT, "data", "nexus-identity.txt"),
    ]
    for data_file in candidates:
        if os.path.exists(data_file):
            ingest_file(data_file)
            return
