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
    context.setdefault("sources", [])
    context.setdefault("visual_artifacts", [])
        
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
    #
    # ROOT CAUSE / KEY FIX:
    # Ollama does NOT populate `tool_call_chunks` on streaming AIMessageChunks.
    # When qwen3 (or any Ollama model) decides to call a tool, it streams the
    # tool-call JSON as plain TEXT content chunks — there is no way to distinguish
    # them from a real text response mid-stream.  Only after the stream ends does
    # Ollama set `tool_calls` on the final AIMessage.
    #
    # Fix: buffer all streamed chunks per LLM invocation (keyed by run_id).
    # On `on_chat_model_end`, inspect the final output:
    #   - output has tool_calls  →  discard buffer (JSON must not reach the UI)
    #   - output has no tool_calls  →  yield the buffer (it is the real answer)
    #
    # Trade-off: the final answer is shown all-at-once instead of token-by-token.
    # This is acceptable because the tool-execution wait already gives visible progress.

    llm_buffers: dict = {}  # run_id -> list[str]

    async for event in graph.astream_events(inputs, version="v2"):
        kind = event["event"]
        run_id = event.get("run_id", "")

        # CRITICAL: Only process LLM events from the "agent" node.
        # Tools (e.g. local_search_tool) invoke their own internal LLMs
        # (query_constructor in _search_with_filter) which also fire
        # on_chat_model_* events. Those produce JSON like
        # {"query": "...", "filter": "NO_FILTER"} that must NOT reach the UI.
        langgraph_node = event.get("metadata", {}).get("langgraph_node")

        if kind == "on_chat_model_start" and langgraph_node == "agent":
            llm_buffers[run_id] = []

        elif kind == "on_chat_model_stream" and langgraph_node == "agent":
            data = event.get("data", {})
            chunk = data.get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                llm_buffers.setdefault(run_id, []).append(chunk.content)

        elif kind == "on_chat_model_end" and langgraph_node == "agent":
            data = event.get("data", {})
            output = data.get("output")
            buffered = llm_buffers.pop(run_id, [])

            # Also skip if this agent LLM invocation was just making a tool call
            has_tool_call = bool(output and getattr(output, "tool_calls", None))

            if not has_tool_call and buffered:
                for text_chunk in buffered:
                    yield text_chunk

        elif kind == "on_tool_end":
            # Capture sources from tool output
            data = event.get("data", {})
            artifact = data.get("artifact")
            output = data.get("output")
            
            print(f"DEBUG TOOL END: Event keys: {data.keys()}")
            print(f"DEBUG TOOL END: Artifact present: {artifact is not None}")
            print(f"DEBUG TOOL END: Output type: {type(output)}")

            metadata = []
            
            # 1. Direct artifact in event data
            if artifact and isinstance(artifact, list):
                print("DEBUG TOOL END: Found artifact in event data")
                metadata = artifact
                
            # 2. Artifact inside ToolMessage object
            elif hasattr(output, 'artifact') and output.artifact and isinstance(output.artifact, list):
                print("DEBUG TOOL END: Found artifact in ToolMessage object")
                metadata = output.artifact
                
            # 3. Legacy Tuple (content, metadata)
            elif isinstance(output, (list, tuple)) and len(output) == 2 and isinstance(output[1], list):
                print("DEBUG TOOL END: Found legacy tuple sources")
                _, metadata = output
            
            if metadata:
                for item in metadata:
                    if not isinstance(item, dict):
                        continue

                    item_type = item.get("type")
                    if item_type == "plot":
                        if item not in context["visual_artifacts"]:
                            context["visual_artifacts"].append(item)
                    else:
                        if item not in context["sources"]:
                            context["sources"].append(item)
                            
    # Final step: append visual artifacts and source citations
    if context.get("visual_artifacts"):
        yield "\n\n"
        for item in context["visual_artifacts"]:
            mime = item.get("mime", "image/png")
            image_base64 = item.get("image_base64", "")
            if image_base64:
                yield f"<nexus-plot mime=\"{mime}\">{image_base64}</nexus-plot>\n"

    if context.get("sources"):
        from urllib.parse import quote
        yield "\n\n### Sources\n"
        for i, src in enumerate(context["sources"], 1):
            title = src.get('title', 'Unknown Title')
            url = src.get('url', '#')
            if not url.startswith('http'):
                url = quote(url, safe='/:')
            yield f"{i}. [{title}]({url})\n"
