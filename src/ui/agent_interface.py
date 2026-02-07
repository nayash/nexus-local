import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.graph import build_graph


# Global cache for the compiled graph
_cached_graph = None

def get_graph():
    global _cached_graph
    if _cached_graph is None:
        print("--- 🏗️ BUILDING GRAPH (Singleton) ---")
        _cached_graph = build_graph()
    return _cached_graph

async def run_agent_stream(query: str, chat_history: list, context: dict = None):
    """
    Runs the agent graph and yields streaming tokens/events.
    
    Args:
        query: The user's latest question.
        chat_history: List of previous messages (formatted for LangChain if needed, 
                      or we can just append the new query here).
        context: Dict containing 'focused_file' or other UI state.
    
    Yields:
        String chunks of the agent's response.
    """
    if context is None:
        context = {}
        
    # 1. Initialize Graph (Cached)
    graph = get_graph()
    
    # 2. Prepare Input State
    # Convert the DB-style chat_history to LangChain message objects
    messages = []
    for msg in chat_history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    
    # Add the current query as the latest HumanMessage
    messages.append(HumanMessage(content=query))
    
    inputs = {
        "messages": messages,
        "focused_file": context.get("focused_file"),
    }
    
    print(f"--- 🚀 LAUNCHING AGENT with query: {query} ---")
    
    # 3. Stream Events
    # We use 'astream_events' to get granular token updates.
    # We assume the LLM is the one generating chunks under 'on_chat_model_stream'.
    
    async for event in graph.astream_events(inputs, version="v2"):
        kind = event["event"]
        
        # Filter for LLM streaming events
        if kind == "on_chat_model_stream":
            # We want to ignore tool calls chunks if possible, or handle them.
            # Usually 'chunk' content is just the text delta.
            data = event["data"]
            if "chunk" in data:
                chunk = data["chunk"]
                # chunk is a BaseMessageChunk (AIMessageChunk)
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content

        elif kind == "on_tool_end":
            # Capture sources from tool output
            # 1. Check for new LangChain 'artifact' field (response_format="content_and_artifact")
            # 2. Fall back to event['output'] tuple (our custom legacy format)
            
            data = event.get("data", {})
            artifact = data.get("artifact")
            output = data.get("output")
            
            metadata = []
            if artifact and isinstance(artifact, list):
                metadata = artifact
            elif isinstance(output, (list, tuple)) and len(output) == 2 and isinstance(output[1], list):
                # It's our legacy (content, metadata) format
                _, metadata = output
            
            if metadata:
                # Accumulate sources
                if "sources" not in context:
                    context["sources"] = []
                
                # Add unique sources
                for item in metadata:
                    if item not in context["sources"]:
                        context["sources"].append(item)
                            
    # Final step: If we have sources, append them to the stream
    if context.get("sources"):
        yield "\n\n### Sources\n"
        for i, src in enumerate(context["sources"], 1):
            title = src.get('title', 'Unknown')
            url = src.get('url', '#')
            # type_ = src.get('type', 'web') # Not displayed but useful for UI
            yield f"{i}. [{title}]({url})\n"
