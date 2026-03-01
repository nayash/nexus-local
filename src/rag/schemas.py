import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_extra(extra: Optional[dict]) -> str:
    return json.dumps(extra or {}, ensure_ascii=True)


def parse_extra(extra) -> dict:
    if isinstance(extra, dict):
        return extra
    if not extra:
        return {}
    try:
        return json.loads(extra)
    except Exception:
        return {}


def build_parent_row(
    *,
    parent_id: str,
    doc_hash: str,
    modality: str,
    text: str,
    source_path: str,
    source_type: str,
    metadata: dict,
    page: Optional[int] = None,
    parent_index: Optional[int] = None,
    image_index: Optional[int] = None,
    mime: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> dict:
    return {
        "parent_id": parent_id,
        "doc_hash": doc_hash,
        "modality": modality,
        "text": text or "",
        "source_path": os.path.abspath(source_path),
        "source_type": source_type,
        "page": page,
        "parent_index": parent_index,
        "image_index": image_index,
        "mime": mime or "",
        "width": width,
        "height": height,
        "indexed_at": utc_now_iso(),
        "extra": json_extra(metadata),
        "workspace_id": metadata.get("workspace_id", "global"),
        "file_name": metadata.get("file_name", os.path.basename(source_path)),
        "file_ext": metadata.get("file_ext", os.path.splitext(source_path)[1].lower()),
        "title": metadata.get("title", os.path.splitext(os.path.basename(source_path))[0]),
        "author": metadata.get("author", ""),
        "owner": metadata.get("owner", "unknown"),
        "document_kind": metadata.get("document_kind", "document"),
        "source_size_bytes": metadata.get("source_size_bytes"),
        "source_mtime_epoch": metadata.get("source_mtime_epoch"),
        "source_mtime_date": metadata.get("source_mtime_date", ""),
        "source_ctime_epoch": metadata.get("source_ctime_epoch"),
        "source_ctime_date": metadata.get("source_ctime_date", ""),
    }


def build_child_row(
    *,
    parent_id: str,
    doc_hash: str,
    vector: list[float],
    embedding_family: str,
    modality: str,
    text: str,
    source_path: str,
    source_type: str,
    metadata: dict,
    page: Optional[int] = None,
    parent_index: Optional[int] = None,
    chunk_index: Optional[int] = None,
    image_index: Optional[int] = None,
    mime: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "parent_id": parent_id,
        "doc_hash": doc_hash,
        "vector": vector,
        "embedding_family": embedding_family,
        "modality": modality,
        "text": text or "",
        "source_path": os.path.abspath(source_path),
        "source_type": source_type,
        "page": page,
        "parent_index": parent_index,
        "chunk_index": chunk_index,
        "image_index": image_index,
        "mime": mime or "",
        "width": width,
        "height": height,
        "indexed_at": utc_now_iso(),
        "extra": json_extra(metadata),
        "workspace_id": metadata.get("workspace_id", "global"),
        "file_name": metadata.get("file_name", os.path.basename(source_path)),
        "file_ext": metadata.get("file_ext", os.path.splitext(source_path)[1].lower()),
        "title": metadata.get("title", os.path.splitext(os.path.basename(source_path))[0]),
        "author": metadata.get("author", ""),
        "owner": metadata.get("owner", "unknown"),
        "document_kind": metadata.get("document_kind", "document"),
        "source_size_bytes": metadata.get("source_size_bytes"),
        "source_mtime_epoch": metadata.get("source_mtime_epoch"),
        "source_mtime_date": metadata.get("source_mtime_date", ""),
        "source_ctime_epoch": metadata.get("source_ctime_epoch"),
        "source_ctime_date": metadata.get("source_ctime_date", ""),
    }


def build_registry_row(
    *,
    source_path: str,
    source_type: str,
    doc_hash: str,
    num_parents: int,
    num_children: int,
    metadata: dict,
) -> dict:
    return {
        "doc_id": doc_hash,
        "source_path": os.path.abspath(source_path),
        "hash": doc_hash,
        "indexed_at": utc_now_iso(),
        "source_type": source_type,
        "num_parents": num_parents,
        "num_children": num_children,
        "workspace_id": metadata.get("workspace_id", "global"),
        "file_name": metadata.get("file_name", os.path.basename(source_path)),
        "file_ext": metadata.get("file_ext", os.path.splitext(source_path)[1].lower()),
        "title": metadata.get("title", os.path.splitext(os.path.basename(source_path))[0]),
        "author": metadata.get("author", ""),
        "owner": metadata.get("owner", "unknown"),
        "document_kind": metadata.get("document_kind", "document"),
        "source_size_bytes": metadata.get("source_size_bytes"),
        "source_mtime_epoch": metadata.get("source_mtime_epoch"),
        "source_mtime_date": metadata.get("source_mtime_date", ""),
        "source_ctime_epoch": metadata.get("source_ctime_epoch"),
        "source_ctime_date": metadata.get("source_ctime_date", ""),
        "extra": json_extra(metadata),
    }
