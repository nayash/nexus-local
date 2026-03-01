from typing import Iterable, Optional

from src.rag.storage import get_db_connection


def get_db():
    return get_db_connection()


def get_or_create_table(table_name: str, sample_rows: Optional[Iterable[dict]] = None):
    db = get_db()
    existing = set(db.table_names())
    if table_name in existing:
        return db.open_table(table_name)

    rows = list(sample_rows or [])
    if not rows:
        return None
    return db.create_table(table_name, rows)


def upsert_rows(table_name: str, rows: list[dict], delete_filter: Optional[str] = None):
    if not rows:
        return None

    db = get_db()
    exists = table_name in set(db.table_names())
    table = get_or_create_table(table_name, rows)
    if table is None:
        return None

    if delete_filter:
        try:
            table.delete(delete_filter)
        except Exception as exc:
            print(f"⚠️ Failed to delete existing rows in '{table_name}': {exc}")

    if exists or delete_filter:
        table.add(rows)
    return table


def search(table_name: str, vector: list[float], filters: Optional[str] = None, top_k: int = 5):
    db = get_db()
    if table_name not in set(db.table_names()):
        return []

    table = db.open_table(table_name)
    query = table.search(vector)
    if filters:
        try:
            query = query.where(filters, prefilter=True)
        except TypeError:
            query = query.where(filters)
    return query.limit(top_k).to_list()


def load_rows(table_name: str) -> list[dict]:
    db = get_db()
    if table_name not in set(db.table_names()):
        return []

    table = db.open_table(table_name)
    try:
        return table.to_list()
    except Exception:
        try:
            frame = table.to_pandas()
            return frame.to_dict(orient="records")
        except Exception:
            return []


def delete_rows(table_name: str, delete_filter: Optional[str] = None):
    db = get_db()
    if table_name not in set(db.table_names()):
        return

    if not delete_filter:
        return

    table = db.open_table(table_name)
    try:
        table.delete(delete_filter)
    except Exception as exc:
        print(f"⚠️ Failed to delete rows in '{table_name}': {exc}")
