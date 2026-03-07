from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.agents.state import AgentState
# from src.agents.nodes import web_search_node, generate_node, local_search_node
from src.agents.nodes import agent_node
from src.agents.router import route_question
from src.tools.registry import TOOLS
from src.tools.tool_results import extract_final_response

# def build_graph_old():
#     """
#     Constructs the branching LangGraph workflow.
#     """
#     workflow = StateGraph(AgentState)
    
#     # 1. Add Nodes
#     workflow.add_node("web_search", web_search_node)
#     workflow.add_node("local_search", local_search_node)
#     workflow.add_node("generate", generate_node)
    
#     # 2. Define Entry Point (The Router)
#     # Instead of pointing to a node, we point to a conditional function
#     workflow.set_conditional_entry_point(
#         route_question,
#         {
#             # Mapper: Output of route_question -> Node Name
#             "web_search": "web_search",
#             "local_search": "local_search",
#             "generate": "generate",
#         }
#     )
    
#     # 3. Define Standard Edges
#     # If we searched, we ALWAYS go to generate afterwards
#     workflow.add_edge("web_search", "generate")
#     workflow.add_edge("local_search", "generate")
    
#     # If we generated, we end
#     workflow.add_edge("generate", END)
    
#     return workflow.compile()

def route_after_tools(state: AgentState):
    messages = state.get("messages") or []
    latest = messages[-1] if messages else None
    if extract_final_response(latest) is not None:
        return END
    return "agent"


def build_graph():
    workflow = StateGraph(AgentState)
    
    # 1. Define Nodes
    workflow.add_node("agent", agent_node)
    
    # ToolNode is a prebuilt worker that executes the function requested by the LLM
    workflow.add_node("tools", ToolNode(TOOLS))
    
    # 2. Define Entry Point
    workflow.set_entry_point("agent")
    
    # 3. Define The Loop (The "ReAct" Pattern)
    
    # After the Agent runs, we check 'tools_condition':
    # - If Agent requested a tool -> Go to 'tools' node
    # - If Agent replied with text -> Go to END
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
    )
    
    # After Tools run, either terminate with the tool-produced final response
    # or loop back to the agent for interpretation.
    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
    )
    
    return workflow.compile()
