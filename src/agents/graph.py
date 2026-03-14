from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.agents.nodes import agent_node
from src.agents.nodes_v2 import (
    identity_worker_node,
    local_catalog_worker_node,
    local_retrieval_worker_node,
    manager_intent_node,
    manager_review_node,
    response_synthesizer_node,
    route_manager_next,
    tabular_worker_node,
    web_retrieval_worker_node,
)
from src.agents.state import AgentState
from src.core.config import Config
from src.tools.registry import TOOLS
from src.tools.tool_results import extract_final_response

def route_after_tools(state: AgentState):
    messages = state.get("messages") or []
    latest = messages[-1] if messages else None
    if extract_final_response(latest) is not None:
        return END
    return "agent"


def build_legacy_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
    )
    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
    )
    return workflow.compile()


def build_manager_v2_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("manager_intent", manager_intent_node)
    workflow.add_node("local_retrieval_worker", local_retrieval_worker_node)
    workflow.add_node("local_catalog_worker", local_catalog_worker_node)
    workflow.add_node("web_retrieval_worker", web_retrieval_worker_node)
    workflow.add_node("identity_worker", identity_worker_node)
    workflow.add_node("tabular_worker", tabular_worker_node)
    workflow.add_node("manager_review", manager_review_node)
    workflow.add_node("response_synthesizer", response_synthesizer_node)

    workflow.set_entry_point("manager_intent")
    workflow.add_conditional_edges(
        "manager_intent",
        route_manager_next,
        {
            "local_retrieval_worker": "local_retrieval_worker",
            "local_catalog_worker": "local_catalog_worker",
            "web_retrieval_worker": "web_retrieval_worker",
            "identity_worker": "identity_worker",
            "tabular_worker": "tabular_worker",
            "response_synthesizer": "response_synthesizer",
        },
    )
    workflow.add_edge("local_retrieval_worker", "manager_review")
    workflow.add_edge("local_catalog_worker", "manager_review")
    workflow.add_edge("web_retrieval_worker", "manager_review")
    workflow.add_edge("identity_worker", "manager_review")
    workflow.add_edge("tabular_worker", "manager_review")
    workflow.add_conditional_edges(
        "manager_review",
        route_manager_next,
        {
            "local_retrieval_worker": "local_retrieval_worker",
            "local_catalog_worker": "local_catalog_worker",
            "web_retrieval_worker": "web_retrieval_worker",
            "identity_worker": "identity_worker",
            "tabular_worker": "tabular_worker",
            "response_synthesizer": "response_synthesizer",
        },
    )
    workflow.add_edge("response_synthesizer", END)
    return workflow.compile()


def build_graph():
    if Config.RAG_PIPELINE_VERSION == "manager_v2":
        return build_manager_v2_graph()
    return build_legacy_graph()
