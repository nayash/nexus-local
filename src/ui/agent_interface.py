import re

from langchain_core.messages import HumanMessage, AIMessage
from src.agents.graph import build_graph


# Global cache for the compiled graph
_cached_graph = None
_TOOL_ARTIFACT_NAMES = {
    "web_search_tool",
    "local_search_tool",
    "get_current_time",
    "analyze_tabular_file_tool",
    "execute_python_code",
}
_TEXT_TOOL_CALL_BLOCK_PATTERN = re.compile(
    r"<\|start_of_tool_call\|>\s*(.*?)\s*<\|end_of_tool_call\|>",
    re.DOTALL,
)
_TEXT_TOOL_CALL_NAME_PATTERN = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(.*\)\s*$", re.DOTALL)


def _looks_like_tool_artifact(text: str) -> bool:
    candidate = (text or "").strip()
    if not candidate:
        return False

    tagged_match = _TEXT_TOOL_CALL_BLOCK_PATTERN.search(candidate)
    if tagged_match:
        candidate = tagged_match.group(1).strip()

    if candidate.startswith("{") and candidate.endswith("}"):
        tool_name_match = re.search(r'"(?:tool|name)"\s*:\s*"([^"]+)"', candidate)
        if tool_name_match and tool_name_match.group(1) in _TOOL_ARTIFACT_NAMES:
            return True

    function_match = _TEXT_TOOL_CALL_NAME_PATTERN.match(candidate)
    if function_match and function_match.group(1) in _TOOL_ARTIFACT_NAMES:
        return True

    return False


def _format_message_for_ui(content: str = "", additional_kwargs: dict | None = None) -> str:
    text_content = content or ""
    if _looks_like_tool_artifact(text_content):
        return ""

    metadata = additional_kwargs or {}
    reasoning_content = (metadata.get("reasoning_content") or "").strip()

    if not reasoning_content:
        return text_content

    if "<think" in text_content.lower():
        return text_content

    wrapped_reasoning = f"<think>{reasoning_content}</think>"
    if text_content.strip():
        return f"{wrapped_reasoning}\n\n{text_content}"
    return wrapped_reasoning


def _create_llm_buffer() -> dict:
    return {
        "saw_stream": False,
        "reasoning_open": False,
    }


def _consume_stream_chunk(buffer_state: dict, chunk) -> list[str]:
    emitted_parts = []
    additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
    reasoning_chunk = (additional_kwargs.get("reasoning_content") or "")
    content_chunk = getattr(chunk, "content", "") or ""

    if reasoning_chunk:
        if not buffer_state["reasoning_open"]:
            emitted_parts.append("<think>")
            buffer_state["reasoning_open"] = True
        emitted_parts.append(reasoning_chunk)

    if content_chunk:
        if buffer_state["reasoning_open"]:
            emitted_parts.append("</think>\n\n")
            buffer_state["reasoning_open"] = False
        emitted_parts.append(content_chunk)

    if emitted_parts:
        buffer_state["saw_stream"] = True

    return emitted_parts


def _close_open_reasoning_block(buffer_state: dict) -> str:
    if buffer_state["reasoning_open"]:
        buffer_state["reasoning_open"] = False
        return "</think>"
    return ""

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
    # The agent now uses a non-streaming tool-decision pass, then a streaming
    # final-answer pass. That means any agent-node stream chunks are safe to send
    # to the UI immediately, while non-stream invocations still need an end-of-run
    # fallback.
    llm_buffers: dict = {}  # run_id -> structured stream state
    pending_nonstream_output = None

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
            pending_nonstream_output = None
            llm_buffers[run_id] = _create_llm_buffer()

        elif kind == "on_chat_model_stream" and langgraph_node == "agent":
            data = event.get("data", {})
            chunk = data.get("chunk")
            if chunk:
                pending_nonstream_output = None
                buffer_state = llm_buffers.setdefault(run_id, _create_llm_buffer())
                for emitted_text in _consume_stream_chunk(buffer_state, chunk):
                    yield emitted_text

        elif kind == "on_chat_model_end" and langgraph_node == "agent":
            data = event.get("data", {})
            output = data.get("output")
            buffer_state = llm_buffers.pop(run_id, _create_llm_buffer())

            # Also skip if this agent LLM invocation was just making a tool call
            has_tool_call = bool(output and getattr(output, "tool_calls", None))

            if not has_tool_call:
                if buffer_state["saw_stream"]:
                    closing_text = _close_open_reasoning_block(buffer_state)
                    if closing_text:
                        yield closing_text
                else:
                    final_content = getattr(output, "content", "") if output else ""
                    final_kwargs = getattr(output, "additional_kwargs", {}) if output else {}
                    rendered = _format_message_for_ui(final_content, final_kwargs)
                    if rendered:
                        pending_nonstream_output = rendered

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
    if pending_nonstream_output:
        yield pending_nonstream_output

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
