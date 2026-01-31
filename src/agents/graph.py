from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from app.agents.state import AgentState
from app.agents.nodes_old import web_search_node, generate_node, local_search_node
from app.agents.nodes import agent_node
from app.agents.router import route_question
from app.tools.registry import TOOLS

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
    
    # After Tools run, ALWAYS go back to Agent to interpret the results
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()