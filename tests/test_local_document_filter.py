from langchain_core.structured_query import Comparator, Comparison, Operation, Operator

from src.tools.local import _normalize_retrieval_plan, _row_matches_filter


def test_like_filter_matches_case_insensitive_filename():
    row = {"file_name": "nexus-local-logs-debug.txt"}
    comparison = Comparison(
        comparator=Comparator.LIKE,
        attribute="file_name",
        value="%NEXUS%",
    )
    assert _row_matches_filter(row, comparison) is True


def test_and_filter_requires_all_filename_terms():
    row = {"file_name": "nexus-local-logs-debug.txt"}
    filter_node = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LIKE, attribute="file_name", value="%nexus%"),
            Comparison(comparator=Comparator.LIKE, attribute="file_name", value="%logs%"),
        ],
    )
    assert _row_matches_filter(row, filter_node) is True


def test_invalid_planner_values_fall_back_to_safe_defaults():
    plan = _normalize_retrieval_plan({"retrieval_mode": "weird", "response_mode": "nope"})
    assert plan == {"retrieval_mode": "semantic_search", "response_mode": "snippets"}
