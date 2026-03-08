from langchain_core.messages import HumanMessage

from src.agents import graph as graph_module
from src.agents.contracts import WorkerResult
from src.agents.nodes_v2 import manager_review_node, tabular_worker_node
from src.tools.schemas import SearchResult
from src.tools.tool_results import build_final_response_artifact


def test_graph_switches_to_manager_v2_when_flag_enabled(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(graph_module.Config, "RAG_PIPELINE_VERSION", "manager_v2")
    monkeypatch.setattr(graph_module, "build_manager_v2_graph", lambda: sentinel)
    assert graph_module.build_graph() is sentinel


def test_graph_switches_to_legacy_when_flag_disabled(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(graph_module.Config, "RAG_PIPELINE_VERSION", "legacy")
    monkeypatch.setattr(graph_module, "build_legacy_graph", lambda: sentinel)
    assert graph_module.build_graph() is sentinel


def test_manager_review_enforces_hop_limit_without_extra_dispatch():
    output = manager_review_node(
        {
            "messages": [HumanMessage(content="find my game ideas")],
            "manager_hop_count": 2,
            "intent_packet": {},
            "current_task": {},
            "task_history": [],
            "evidence_bundle": {},
        }
    )
    assert output["manager_hop_count"] == 3
    assert output["manager_next_node"] == "response_synthesizer"


def test_execute_local_retrieval_task_v2_hybrid_combines_semantic_and_inventory(monkeypatch):
    from src.tools import local as local_module

    def fake_semantic(query, file_filter=None, top_k=10, apply_lexical_supplement=True):
        assert apply_lexical_supplement is False
        return [
            SearchResult(
                title="Local File (md): writing_ideas.md",
                url="/tmp/writing_ideas.md",
                content="Core idea: a text-based adventure with memory puzzles.",
                source="local",
            )
        ]

    monkeypatch.setattr(local_module, "_search_multimodal_results", fake_semantic)
    monkeypatch.setattr(
        local_module,
        "_resolve_direct_local_response_v2",
        lambda query, file_filter="", explicit_mode="document_lookup": (
            "",
            [
                {"title": "Local File (md): writing_ideas.md", "url": "/tmp/writing_ideas.md", "type": "local"},
                build_final_response_artifact("Matching files (1):\n- writing_ideas.md :: /tmp/writing_ideas.md"),
            ],
        ),
    )

    payload = local_module.execute_local_retrieval_task_v2(
        query="look for the game ideas I had in my files",
        mode="hybrid",
    )
    result = WorkerResult.model_validate(payload)
    assert result.status == "ok"
    assert "Core idea" in result.summary
    assert "Matching files (1)" in result.summary
    assert result.source_metadata


def test_search_local_routes_to_v2_when_pipeline_enabled(monkeypatch):
    from src.tools import local as local_module

    monkeypatch.setattr(local_module.Config, "RAG_PIPELINE_VERSION", "manager_v2")
    monkeypatch.setattr(
        local_module,
        "_search_local_v2",
        lambda query, file_filter="": [
            SearchResult(title="Info", url="", content="v2 path", source="local")
        ],
    )
    monkeypatch.setattr(local_module, "_search_local_legacy", lambda *args, **kwargs: [])

    output = local_module.search_local("find my notes")
    assert output[0].content == "v2 path"


def test_tabular_worker_preserves_plot_metadata_for_ui(monkeypatch):
    from src.agents import nodes_v2 as nodes_v2_module

    monkeypatch.setattr(
        nodes_v2_module.analyze_tabular_file_tool,
        "func",
        lambda file_path, user_query: (
            "tabular content",
            [
                {"title": "sales_data_dummy.csv", "url": file_path, "type": "local"},
                {
                    "type": "plot",
                    "mime": "image/png",
                    "image_base64": "plotb64",
                    "summary": "Revenue rises over time.",
                    "title": "Plot for sales_data_dummy.csv",
                },
            ],
        ),
    )

    result = tabular_worker_node(
        {
            "messages": [HumanMessage(content="Generate a plot and show here for Revenue vs Date")],
            "focused_file": "/tmp/sales_data_dummy.csv",
            "current_task": {"query": "Generate a plot and show here for Revenue vs Date"},
            "evidence_bundle": {},
            "task_history": [],
        }
    )

    sources = result["evidence_bundle"]["source_metadata"]
    assert any(item.get("type") == "plot" for item in sources)
