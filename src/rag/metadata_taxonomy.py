import re
from typing import Optional


CANONICAL_DOCUMENT_KINDS = (
    "book",
    "log",
    "data",
    "image",
    "document",
    "code",
)

_DOCUMENT_KIND_ALIASES = {
    "book": "book",
    "books": "book",
    "novel": "book",
    "novels": "book",
    "story": "book",
    "stories": "book",
    "poem": "book",
    "poems": "book",
    "log": "log",
    "logs": "log",
    "logging": "log",
    "data": "data",
    "dataset": "data",
    "datasets": "data",
    "table": "data",
    "tables": "data",
    "spreadsheet": "data",
    "spreadsheets": "data",
    "image": "image",
    "images": "image",
    "photo": "image",
    "photos": "image",
    "picture": "image",
    "pictures": "image",
    "screenshot": "image",
    "screenshots": "image",
    "document": "document",
    "documents": "document",
    "doc": "document",
    "docs": "document",
    "file": "document",
    "files": "document",
    "note": "document",
    "notes": "document",
    "notebook": "document",
    "notebooks": "document",
    "memo": "document",
    "memos": "document",
    "journal": "document",
    "journals": "document",
    "code": "code",
    "script": "code",
    "scripts": "code",
}

_GENERIC_CORPUS_PHRASES = (
    "my files",
    "my file",
    "my docs",
    "my doc",
    "my documents",
    "my notes",
    "my note",
    "from my files",
    "from my docs",
    "from my documents",
    "from my notes",
)


def normalize_document_kind(value: Optional[str], default: Optional[str] = None) -> Optional[str]:
    if value is None:
        return default

    normalized = re.sub(r"[_\-\s]+", " ", str(value).strip().lower())
    if not normalized:
        return default
    return _DOCUMENT_KIND_ALIASES.get(normalized, default)


def is_valid_document_kind(value: Optional[str]) -> bool:
    normalized = normalize_document_kind(value)
    return bool(normalized and normalized in CANONICAL_DOCUMENT_KINDS)


def infer_document_kind_from_query(query: str) -> Optional[str]:
    lowered = f" {query.lower()} "

    for token in ("log", "logs"):
        if f" {token} " in lowered:
            return "log"
    for token in ("book", "books", "novel", "novels"):
        if f" {token} " in lowered:
            return "book"
    for token in ("image", "images", "photo", "photos", "screenshot", "screenshots", "picture", "pictures"):
        if f" {token} " in lowered:
            return "image"
    for token in ("csv", "dataset", "datasets", "table", "tables", "spreadsheet", "spreadsheets", "data"):
        if f" {token} " in lowered:
            return "data"
    for token in ("script", "scripts", "code"):
        if f" {token} " in lowered:
            return "code"
    for token in (
        "document",
        "documents",
        "doc",
        "docs",
        "file",
        "files",
        "notes",
        "note",
        "notebook",
        "notebooks",
        "memo",
        "memos",
        "journal",
        "journals",
    ):
        if f" {token} " in lowered:
            return "document"
    return None


def is_broad_personal_corpus_query(query: str) -> bool:
    lowered = query.lower()
    return any(phrase in lowered for phrase in _GENERIC_CORPUS_PHRASES)
