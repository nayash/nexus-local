import re
from dataclasses import dataclass
from typing import Optional, Tuple
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)

class LanceDBTranslator(Visitor):
    """Custom Translator to convert LangChain structured queries to LanceDB SQL filters."""
    
    allowed_operators = [Operator.AND, Operator.OR, Operator.NOT]
    allowed_comparators = [
        Comparator.EQ,
        Comparator.NE,
        Comparator.GT,
        Comparator.GTE,
        Comparator.LT,
        Comparator.LTE,
        Comparator.IN,
        Comparator.LIKE,
    ]

    def _format_func(self, func: Operator | Comparator) -> str:
        return func.value

    def visit_operation(self, operation: Operation) -> str:
        args = [arg.accept(self) for arg in operation.arguments]
        operator = self._format_func(operation.operator)
        if operator == "not":
            return f"NOT ({args[0]})"
        return f" {operator.upper()} ".join(f"({arg})" for arg in args)

    def visit_comparison(self, comparison: Comparison) -> str:
        attr = comparison.attribute
        val = comparison.value
        comparator = self._format_func(comparison.comparator)

        # Format string values with quotes for SQL
        if isinstance(val, str):
            val = f"'{val}'"
        elif isinstance(val, list):
            # Format lists for SQL IN clauses
            val = f"({', '.join([f'{repr(v)}' for v in val])})"

        # Map LangChain comparators to SQL syntax
        if comparator == "eq":
            return f"{attr} = {val}"
        elif comparator == "ne":
            return f"{attr} != {val}"
        elif comparator == "gt":
            return f"{attr} > {val}"
        elif comparator == "gte":
            return f"{attr} >= {val}"
        elif comparator == "lt":
            return f"{attr} < {val}"
        elif comparator == "lte":
            return f"{attr} <= {val}"
        elif comparator == "in":
            return f"{attr} IN {val}"
        elif comparator == "like":
            return f"{attr} LIKE {val}"
        else:
            raise NotImplementedError(f"Unknown comparator: {comparator}")

    def visit_structured_query(self, structured_query: StructuredQuery) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"filter": structured_query.filter.accept(self)}
        return structured_query.query or "", kwargs


# ---------------------------------------------------------------------------
# Schema-aware filter resolver
# ---------------------------------------------------------------------------

_SQL_OPERATORS = r'(?:=|!=|>|<|>=|<=|\bIN\b|\bLIKE\b)'
_COLUMN_PATTERN = re.compile(rf'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*{_SQL_OPERATORS}', re.IGNORECASE)


@dataclass
class FilterResolution:
    """Result of resolving a SQL filter string against a LanceDB table schema."""
    filter: Optional[str]   # Final SQL filter to pass to LanceDB (None → skip)
    applicable: bool        # Whether the filter should be applied
    rewrites: list          # Columns that were rewritten to metadata.col
    skipped: list           # Columns that were absent from both top-level and struct


def resolve_lancedb_filter(native_filter: str, schema) -> FilterResolution:
    """
    Validate and optionally rewrite a SQL filter string against a LanceDB table schema.

    Two ingestion strategies produce different schemas:
    - Naive: flat columns  (vector, text, source, id)
    - Parent-child: all user metadata packed inside a 'metadata' struct column

    This function:
    1. Checks each column referenced in the filter against top-level schema fields.
    2. If a column is missing at the top level but present inside the 'metadata' struct,
       rewrites it in-place: ``author = 'x'`` → ``metadata.author = 'x'``.
    3. Returns a FilterResolution describing what happened.
       - ``applicable=True``  → use the (possibly rewritten) filter.
       - ``applicable=False`` → one or more columns are genuinely absent; skip the filter.

    Args:
        native_filter: SQL filter string produced by LanceDBTranslator.
        schema:        PyArrow schema from ``vectorstore._table.schema``.

    Returns:
        FilterResolution dataclass.
    """
    import pyarrow as pa

    top_level_cols: set = set(schema.names)

    # Collect sub-field names from the 'metadata' struct column (if it exists)
    metadata_sub_cols: set = set()
    if "metadata" in top_level_cols:
        metadata_field = schema.field("metadata")
        if pa.types.is_struct(metadata_field.type):
            metadata_sub_cols = {
                metadata_field.type.field(i).name
                for i in range(metadata_field.type.num_fields)
            }

    referenced_cols = _COLUMN_PATTERN.findall(native_filter)

    rewrites: list = []
    skipped: list = []
    rewritten_filter = native_filter

    for col in referenced_cols:
        if col in top_level_cols:
            continue  # column directly available — no rewrite needed

        if col in metadata_sub_cols:
            # Rewrite: bare `col` → `metadata.col`
            rewritten_filter = re.sub(
                rf'\b{re.escape(col)}\b(?=\s*{_SQL_OPERATORS})',
                f'metadata.{col}',
                rewritten_filter,
                flags=re.IGNORECASE,
            )
            rewrites.append(col)
        else:
            skipped.append(col)

    applicable = len(skipped) == 0
    return FilterResolution(
        filter=rewritten_filter if applicable else None,
        applicable=applicable,
        rewrites=rewrites,
        skipped=skipped,
    )
