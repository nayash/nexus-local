from langchain_core.structured_query import Comparator, Comparison, Operation, Operator

from src.tools.local import _normalize_retrieval_plan, _row_matches_filter
from src.tools.local import _rerank_and_filter_rows
from src.rag.query_filters import compile_multimodal_filter, compile_multimodal_filter_plan


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


def test_schema_lock_normalizes_note_kind_from_planner(monkeypatch):
    class StubStructuredQuery:
        def __init__(self):
            self.query = "writing ideas"
            self.filter = Comparison(
                comparator=Comparator.EQ,
                attribute="document_kind",
                value="note",
            )

    monkeypatch.setattr("src.rag.query_filters._get_query_constructor", lambda: type("X", (), {"invoke": lambda *_: StubStructuredQuery()})())
    _, sql_filter = compile_multimodal_filter("Give me writing ideas from my notes")
    assert sql_filter == "document_kind = 'document'"


def test_plan_relaxes_low_confidence_filters_for_broad_personal_queries(monkeypatch):
    class StubStructuredQuery:
        def __init__(self):
            self.query = "writing ideas"
            self.filter = Comparison(
                comparator=Comparator.EQ,
                attribute="document_kind",
                value="document",
            )

    monkeypatch.setattr("src.rag.query_filters._get_query_constructor", lambda: type("X", (), {"invoke": lambda *_: StubStructuredQuery()})())
    plan = compile_multimodal_filter_plan("Give me writing ideas from my notes")
    assert plan.strict_sql_filter == "document_kind = 'document'"
    assert plan.should_try_relaxed is True
    assert plan.relaxed_sql_filter is None


def test_reranker_prioritizes_relevant_note_and_deduplicates_same_source():
    rows = [
        {
            "source_path": "/tmp/NestedLearning.pdf",
            "source_type": "pdf",
            "file_name": "NestedLearning.pdf",
            "title": "NestedLearning",
            "text": "associative memory and gradient descent",
            "extra": "{}",
            "_score": 0.95,
        },
        {
            "source_path": "/tmp/writing_tips.md",
            "source_type": "md",
            "file_name": "writing_tips.md",
            "title": "Writing tips",
            "text": "Writing tips for story flow and character consistency",
            "extra": "{}",
            "_score": 0.70,
        },
        {
            "source_path": "/tmp/NestedLearning.pdf",
            "source_type": "pdf",
            "file_name": "NestedLearning.pdf",
            "title": "NestedLearning",
            "text": "optimization formulation and equations",
            "extra": "{}",
            "_score": 0.80,
        },
    ]
    ranked = _rerank_and_filter_rows(rows, "Give me writing tips from my notes", top_k=5)
    assert ranked
    assert ranked[0]["source_path"] == "/tmp/writing_tips.md"
    assert len({row["source_path"] for row in ranked}) == len(ranked)
