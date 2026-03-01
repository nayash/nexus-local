import os
from typing import Callable, Literal, Optional

from src.rag.ingestion_multimodal import SUPPORTED_MULTIMODAL_EXTENSIONS, ingest_file_multimodal


def ingest_file(
    file_path: str,
    table_name: str = "",
    strategy: Literal["multimodal"] = "multimodal",
):
    """
    Ingest a single file using the shared multimodal pipeline.

    `table_name` is accepted for compatibility with older call sites but ignored.
    """
    _ = table_name, strategy
    return ingest_file_multimodal(file_path)


def ingest_path(
    path: str,
    strategy: Literal["multimodal"] = "multimodal",
    progress_callback: Optional[Callable] = None,
):
    """
    Ingest a file or directory using the shared multimodal pipeline.
    """
    _ = strategy
    print(f"ingesting path {path} with strategy multimodal")

    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        return ingest_file_multimodal(path, progress_callback=progress_callback)

    if os.path.isdir(path):
        files_to_ingest = []
        for root, _, files in os.walk(path):
            for file_name in files:
                if file_name.lower().endswith(SUPPORTED_MULTIMODAL_EXTENSIONS):
                    files_to_ingest.append(os.path.join(root, file_name))

        if not files_to_ingest:
            return False, "No supported files found.", None

        total_chunks = 0
        successful_files = 0
        for file_path in files_to_ingest:
            success, chunk_count, _ = ingest_file_multimodal(file_path, progress_callback=progress_callback)
            if success:
                successful_files += 1
                total_chunks += chunk_count

        return True, f"Successfully ingested {successful_files} files ({total_chunks} chunks).", None

    return False, "Path does not exist.", None


def init_knowledge():
    """
    Seed the local knowledge base with the Nexus identity file when present.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    data_file = os.path.join(project_root, "data", "nexus-identity.txt")
    if os.path.exists(data_file):
        ingest_file(data_file)
