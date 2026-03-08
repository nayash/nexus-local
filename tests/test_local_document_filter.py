from langchain_core.structured_query import Comparator, Comparison, Operation, Operator
import pytest

from src.tools.local import _normalize_retrieval_plan, _row_matches_filter
from src.tools.local import _rerank_and_filter_rows, _search_multimodal_results, _should_apply_lexical_supplement, resolve_direct_local_response
from src.rag.query_filters import compile_multimodal_filter, compile_multimodal_filter_plan
from src.tools.tool_results import extract_final_response


@pytest.fixture(autouse=True)
def _force_legacy_pipeline(monkeypatch):
    monkeypatch.setattr("src.tools.local.Config.RAG_PIPELINE_VERSION", "legacy")


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


def test_corrective_prompt_builds_precise_focus_filter_and_blocks_unfiltered_fallback(monkeypatch):
    monkeypatch.setattr("src.rag.query_filters._get_query_constructor", lambda: (_ for _ in ()).throw(RuntimeError("no llm")))
    monkeypatch.setattr(
        "src.rag.query_filters._llm_normalize_query",
        lambda query: ('list down the "Writing tips" from my files', ["Writing tips"], "high", True),
    )

    plan = compile_multimodal_filter_plan(
        'why are you giving me prompts and ideas?? I asked you to list down the "Writing tips" from my files.'
    )

    strict_sql = (plan.strict_sql_filter or "").lower()
    assert plan.text_query == 'list down the "Writing tips" from my files'
    assert "writing tips" in strict_sql
    assert "title like" in strict_sql or "file_name like" in strict_sql or "source_path like" in strict_sql
    assert plan.allow_unfiltered_fallback is False


def test_deterministic_corrective_rewrite_runs_when_llm_normalizer_fails(monkeypatch):
    monkeypatch.setattr("src.rag.query_filters._get_query_constructor", lambda: (_ for _ in ()).throw(RuntimeError("no llm")))
    monkeypatch.setattr("src.rag.query_filters._llm_normalize_query", lambda query: (query, [], "low", False))

    plan = compile_multimodal_filter_plan(
        'why are you giving me prompts and ideas?? I asked you to list down the "Writing tips" from my files.'
    )

    assert plan.text_query.lower().startswith("list down the")


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


def test_explicit_file_inventory_query_forces_document_lookup_even_when_planner_says_semantic(monkeypatch):
    monkeypatch.setattr(
        "src.tools.local._plan_local_retrieval",
        lambda query, file_filter="": {"retrieval_mode": "semantic_search", "response_mode": "snippets"},
    )
    monkeypatch.setattr(
        "src.tools.local._query_documents_table",
        lambda query, file_filter="": [
            {
                "title": "Text Based Adventure - Abaddon Hotel",
                "file_name": "abaddon.md",
                "source_path": "/tmp/abaddon.md",
                "source_type": "md",
            }
        ],
    )

    payload = resolve_direct_local_response("Do I have any files about text based adventure game ideas?")
    assert payload is not None
    response = extract_final_response(payload)
    assert response is not None
    assert "Yes, I found 1 matching idea file(s):" in response
    assert "Abaddon Hotel" in response


def test_content_query_does_not_short_circuit_to_file_listing(monkeypatch):
    monkeypatch.setattr(
        "src.tools.local._plan_local_retrieval",
        lambda query, file_filter="": {"retrieval_mode": "document_lookup", "response_mode": "snippets"},
    )
    monkeypatch.setattr(
        "src.tools.local._query_documents_table",
        lambda query, file_filter="": [
            {
                "title": "Writing ideas",
                "file_name": "writing_ideas.md",
                "source_path": "/tmp/writing_ideas.md",
                "source_type": "md",
            }
        ],
    )

    payload = resolve_direct_local_response("List down the writing ideas I had")
    assert payload is None


def test_single_match_snippet_query_defers_to_semantic_when_not_metadata(monkeypatch):
    monkeypatch.setattr(
        "src.tools.local._plan_local_retrieval",
        lambda query, file_filter="": {"retrieval_mode": "document_lookup", "response_mode": "snippets"},
    )
    monkeypatch.setattr(
        "src.tools.local._query_documents_table",
        lambda query, file_filter="": [
            {
                "title": "Project report",
                "file_name": "project_report.md",
                "source_path": "/tmp/project_report.md",
                "source_type": "md",
            }
        ],
    )

    payload = resolve_direct_local_response("Summarize the project report")
    assert payload is None


def test_single_match_file_location_query_returns_path(monkeypatch):
    monkeypatch.setattr(
        "src.tools.local._plan_local_retrieval",
        lambda query, file_filter="": {"retrieval_mode": "document_lookup", "response_mode": "snippets"},
    )
    monkeypatch.setattr(
        "src.tools.local._query_documents_table",
        lambda query, file_filter="": [
            {
                "title": "Project report",
                "file_name": "project_report.md",
                "source_path": "/tmp/project_report.md",
                "source_type": "md",
            }
        ],
    )

    payload = resolve_direct_local_response("Which file is the project report?")
    response = extract_final_response(payload)
    assert response == "Matched file:\n/tmp/project_report.md"


def test_multimodal_search_supplements_lexical_parent_rows(monkeypatch):
    semantic_rows = [
        {
            "source_path": "/tmp/other_note.md",
            "source_type": "md",
            "file_name": "other_note.md",
            "title": "Other Note",
            "text": "general productivity notes",
            "extra": "{}",
            "_score": 0.2,
            "modality": "text",
            "page": 1,
            "parent_index": 1,
        }
    ]
    lexical_docs = [
        {
            "source_path": "/tmp/abaddon.md",
            "source_type": "md",
            "file_name": "abaddon.md",
            "title": "Text Based Adventure - Abaddon Hotel",
            "_lexical_score": 1.4,
        }
    ]
    parent_rows = [
        {
            "source_path": "/tmp/abaddon.md",
            "source_type": "md",
            "file_name": "abaddon.md",
            "title": "Text Based Adventure - Abaddon Hotel",
            "text": "Abaddon Hotel is a text based adventure game concept.",
            "extra": "{}",
            "modality": "text",
            "page": 1,
            "parent_index": 0,
        }
    ]

    monkeypatch.setattr("src.tools.local.search_multimodal", lambda *args, **kwargs: semantic_rows)
    monkeypatch.setattr("src.tools.local._lexical_document_candidates", lambda *args, **kwargs: lexical_docs)
    monkeypatch.setattr("src.tools.local.load_rows", lambda table_name: parent_rows)

    results = _search_multimodal_results("Did I have any text based adventure game idea?", top_k=5)
    assert any(result.url == "/tmp/abaddon.md" for result in results)


def test_lexical_supplement_intent_guard():
    assert _should_apply_lexical_supplement("Did I have any text based adventure game idea?") is True
    assert _should_apply_lexical_supplement("screenshot of the error message") is False
