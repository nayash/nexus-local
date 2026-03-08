from typing import Iterable, Optional

from src.rag.storage import get_db_connection


_INT_SENTINEL_COLUMNS = {
    "page": -1,
    "parent_index": -1,
    "chunk_index": -1,
    "image_index": -1,
    "width": 0,
    "height": 0,
    "source_size_bytes": 0,
    "source_mtime_epoch": 0,
    "source_ctime_epoch": 0,
    "num_parents": 0,
    "num_children": 0,
}

_OPTIONAL_INT_SENTINELS = {
    "page": -1,
    "parent_index": -1,
    "chunk_index": -1,
    "image_index": -1,
}


def get_db():
    return get_db_connection()


def list_tables(db) -> set[str]:
    def _coerce_name(item):
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("name", "table_name"):
                value = item.get(key)
                if isinstance(value, str):
                    return value
        if isinstance(item, (list, tuple)):
            for value in item:
                coerced = _coerce_name(value)
                if coerced:
                    return coerced
        return None

    try:
        raw_tables = db.list_tables()
    except AttributeError:
        raw_tables = db.table_names()

    names = set()
    for item in raw_tables:
        name = _coerce_name(item)
        if name:
            names.add(name)
    return names


def _normalize_row_types(rows: Iterable[dict]) -> list[dict]:
    normalized_rows = []
    for row in rows:
        normalized = dict(row)
        for key, default in _INT_SENTINEL_COLUMNS.items():
            if key in normalized and normalized[key] is None:
                normalized[key] = default
        normalized_rows.append(normalized)
    return normalized_rows


def _denormalize_row_types(rows: Iterable[dict]) -> list[dict]:
    denormalized_rows = []
    for row in rows:
        denormalized = dict(row)
        for key, sentinel in _OPTIONAL_INT_SENTINELS.items():
            if denormalized.get(key) == sentinel:
                denormalized[key] = None
        denormalized_rows.append(denormalized)
    return denormalized_rows


def get_or_create_table(table_name: str, sample_rows: Optional[Iterable[dict]] = None):
    db = get_db()
    try:
        return db.open_table(table_name)
    except Exception:
        pass

    rows = _normalize_row_types(sample_rows or [])
    if not rows:
        return None
    return db.create_table(table_name, rows)


def _rebuild_table_with_rows(db, table_name: str, rows: list[dict]):
    normalized_rows = _normalize_row_types(rows)
    db.drop_table(table_name)
    return db.create_table(table_name, normalized_rows)


def upsert_rows(table_name: str, rows: list[dict], delete_filter: Optional[str] = None):
    if not rows:
        return None

    rows = _normalize_row_types(rows)
    db = get_db()
    try:
        table = db.open_table(table_name)
        exists = True
    except Exception:
        exists = False
        table = get_or_create_table(table_name, rows)
        if table is None:
            return None

    if delete_filter:
        try:
            table.delete(delete_filter)
        except Exception as exc:
            print(f"⚠️ Failed to delete existing rows in '{table_name}': {exc}")

    if exists or delete_filter:
        try:
            table.add(rows)
        except Exception as exc:
            message = str(exc)
            if "cast_null" in message or "Unsupported cast from int64 to null" in message:
                print(f"⚠️ Rebuilding '{table_name}' due to incompatible null-typed schema")
                existing_rows = load_rows(table_name)
                return _rebuild_table_with_rows(db, table_name, existing_rows + rows)
            raise
    return table


def search(table_name: str, vector: list[float], filters: Optional[str] = None, top_k: int = 5):
    db = get_db()
    try:
        table = db.open_table(table_name)
    except Exception:
        return []
    query = table.search(vector)
    if filters:
        try:
            query = query.where(filters, prefilter=True)
        except TypeError:
            query = query.where(filters)
    return _denormalize_row_types(query.limit(top_k).to_list())


def load_rows(table_name: str) -> list[dict]:
    db = get_db()
    try:
        table = db.open_table(table_name)
    except Exception:
        return []
    try:
        return _denormalize_row_types(table.to_list())
    except Exception:
        try:
            frame = table.to_pandas()
            return _denormalize_row_types(frame.to_dict(orient="records"))
        except Exception:
            return []


def delete_rows(table_name: str, delete_filter: Optional[str] = None):
    db = get_db()
    try:
        table = db.open_table(table_name)
    except Exception:
        return

    if not delete_filter:
        return
    try:
        table.delete(delete_filter)
    except Exception as exc:
        print(f"⚠️ Failed to delete rows in '{table_name}': {exc}")
