import re
import uuid

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.agents.graph import build_graph
from src.tools.tool_results import extract_artifacts, find_final_response_artifact


# Global cache for the compiled graph
_cached_graph = None
_TOOL_ARTIFACT_NAMES = {
    "web_search_tool",
    "local_search_tool",
    "lookup_local_files_tool",
    "get_nexus_identity_tool",
    "get_current_time",
    "analyze_tabular_file_tool",
    "execute_python_code",
}
_TEXT_TOOL_CALL_BLOCK_PATTERN = re.compile(
    r"<\|start_of_tool_call\|>\s*(.*?)\s*<\|end_of_tool_call\|>",
    re.DOTALL,
)
_TEXT_TOOL_CALL_NAME_PATTERN = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(.*\)\s*$", re.DOTALL)
_REASONING_FOLLOWUP_PATTERNS = (
    r"\b(reasoning|thought process|thinking|why did you answer|why this answer)\b",
    r"\b(explain (your|the) (reasoning|thinking))\b",
    r"\b(how did you (arrive|come) at)\b",
)
_RESPONSE_PHASE_TAG_PREFIX = "nexus_phase:"
_RESPONSE_PHASE_DECISION = "decision"
_VISIBLE_RESPONSE_PHASES = {"final", "final_retry"}


def _strip_think_markup_for_ui(text: str) -> str:
    """
    Remove inline <think> markup from text destined for visible UI content.
    If a think block is unclosed, hide everything after the opening tag.
    """
    raw = text or ""
    if not raw:
        return ""

    lower = raw.lower()
    out_parts = []
    cursor = 0
    close_tag = "</think>"

    while cursor < len(raw):
        open_idx = lower.find("<think", cursor)
        close_idx = lower.find(close_tag, cursor)

        if open_idx == -1 and close_idx == -1:
            out_parts.append(raw[cursor:])
            break

        if close_idx != -1 and (open_idx == -1 or close_idx < open_idx):
            out_parts.append(raw[cursor:close_idx])
            cursor = close_idx + len(close_tag)
            continue

        out_parts.append(raw[cursor:open_idx])
        open_end = lower.find(">", open_idx)
        if open_end == -1:
            break

        close_after_open = lower.find(close_tag, open_end + 1)
        if close_after_open == -1:
            break

        cursor = close_after_open + len(close_tag)

    return "".join(out_parts).strip()


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
    return _strip_think_markup_for_ui(text_content)


def _is_reasoning_followup_query(query: str) -> bool:
    normalized = " ".join((query or "").strip().lower().split())
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in _REASONING_FOLLOWUP_PATTERNS)


def _create_llm_buffer(phase: str = "") -> dict:
    return {
        "saw_stream": False,
        "reasoning_parts": [],
        "content_parts": [],
        "phase": (phase or "").strip().lower(),
        "ui_emitted": False,
        "stream_chunk_count": 0,
        "stream_char_count": 0,
    }


def _extract_response_phase(event: dict) -> str:
    metadata = event.get("metadata", {}) or {}
    phase = metadata.get("nexus_phase") or metadata.get("response_phase") or ""
    if isinstance(phase, str) and phase.strip():
        return phase.strip().lower()

    for tag in event.get("tags", []) or []:
        if not isinstance(tag, str):
            continue
        if tag.startswith(_RESPONSE_PHASE_TAG_PREFIX):
            return tag[len(_RESPONSE_PHASE_TAG_PREFIX):].strip().lower()

    return ""


def _consume_stream_chunk(buffer_state: dict, chunk) -> list[str]:
    emitted_parts = []
    additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
    reasoning_chunk = (additional_kwargs.get("reasoning_content") or "")
    content_chunk = getattr(chunk, "content", "") or ""

    if reasoning_chunk:
        buffer_state["reasoning_parts"].append(reasoning_chunk)

    if content_chunk:
        emitted_parts.append(content_chunk)
        buffer_state["content_parts"].append(content_chunk)

    if emitted_parts:
        buffer_state["saw_stream"] = True

    return emitted_parts


def _collect_reasoning_text(buffer_state: dict, output) -> str:
    streamed = "".join(buffer_state.get("reasoning_parts", [])).strip()
    if streamed:
        return streamed
    metadata = getattr(output, "additional_kwargs", {}) if output else {}
    return (metadata.get("reasoning_content") or "").strip()


def _phase_allows_ui_stream(phase: str) -> bool:
    return (phase or "") in _VISIBLE_RESPONSE_PHASES


def _phase_allows_nonstream_output(phase: str) -> bool:
    normalized = (phase or "").strip().lower()
    if normalized == _RESPONSE_PHASE_DECISION:
        return False
    # Unknown phase falls back to "allow"; final arbitration keeps only the last candidate.
    return True

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
    context.setdefault("last_turn_reasoning", "")
        
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
    
    # Inject previous-turn reasoning summary as private context when the user asks
    # for reasoning about the last answer. Keep this separate from visible content.
    previous_reasoning = (context.get("last_assistant_reasoning") or "").strip()
    if previous_reasoning and _is_reasoning_followup_query(query):
        messages.append(
            SystemMessage(
                content=(
                    "Previous assistant-turn reasoning summary (private context):\n"
                    f"{previous_reasoning}\n\n"
                    "If the user asks how/why the last answer was produced, explain from this summary."
                )
            )
        )

    # Add the current query as the latest HumanMessage
    messages.append(HumanMessage(content=query))
    
    inputs = {
        "messages": messages,
        "focused_file": context.get("focused_file"),
    }
    trace_id = uuid.uuid4().hex[:8]
    query_preview = " ".join((query or "").split())[:180]
    print(
        f"[RAG {trace_id}] start | history_msgs={len(chat_history)} | "
        f'focused_file={context.get("focused_file")} | query="{query_preview}"'
    )
    
    # 3. Stream Events
    # The agent now uses a non-streaming tool-decision pass, then a streaming
    # final-answer pass. That means any agent-node stream chunks are safe to send
    # to the UI immediately, while non-stream invocations still need an end-of-run
    # fallback.
    llm_buffers: dict = {}  # run_id -> structured stream state
    pending_nonstream_output = None
    fallback_nonstream_output = None
    last_reasoning_snapshot = ""
    answer_emitted = False

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
            phase = _extract_response_phase(event)
            llm_buffers[run_id] = _create_llm_buffer(phase=phase)
            print(f"[RAG {trace_id}] llm_start | run_id={run_id[:8]} | phase={phase or 'unknown'}")

        elif kind == "on_chat_model_stream" and langgraph_node == "agent":
            data = event.get("data", {})
            chunk = data.get("chunk")
            if chunk:
                pending_nonstream_output = None
                buffer_state = llm_buffers.setdefault(run_id, _create_llm_buffer())
                phase = buffer_state.get("phase", "")
                should_emit_stream = _phase_allows_ui_stream(phase)
                for emitted_text in _consume_stream_chunk(buffer_state, chunk):
                    buffer_state["stream_chunk_count"] += 1
                    buffer_state["stream_char_count"] += len(emitted_text)
                    if should_emit_stream:
                        if buffer_state["stream_chunk_count"] == 1:
                            print(
                                f"[RAG {trace_id}] llm_stream_begin | "
                                f"run_id={run_id[:8]} | phase={phase or 'unknown'}"
                            )
                        yield emitted_text
                        answer_emitted = True
                        buffer_state["ui_emitted"] = True

        elif kind == "on_chat_model_end" and langgraph_node == "agent":
            data = event.get("data", {})
            output = data.get("output")
            buffer_state = llm_buffers.pop(run_id, _create_llm_buffer())
            phase = buffer_state.get("phase", "")

            # Also skip if this agent LLM invocation was just making a tool call
            has_tool_call = bool(output and getattr(output, "tool_calls", None))
            reasoning_snapshot = _collect_reasoning_text(buffer_state, output)
            if reasoning_snapshot and _phase_allows_nonstream_output(phase):
                last_reasoning_snapshot = reasoning_snapshot
            content_len = len((getattr(output, "content", "") or "")) if output else 0
            tool_call_count = len(getattr(output, "tool_calls", []) or []) if output else 0
            print(
                f"[RAG {trace_id}] llm_end | run_id={run_id[:8]} | phase={phase or 'unknown'} | "
                f"tool_calls={tool_call_count} | content_chars={content_len} | "
                f"stream_chunks={buffer_state.get('stream_chunk_count', 0)} | "
                f"ui_emitted={buffer_state.get('ui_emitted', False)}"
            )

            if not has_tool_call:
                # If no UI text was emitted for this invocation, convert the final
                # content to a candidate answer. This catches non-streaming outputs
                # and intentionally-suppressed stream phases.
                if not buffer_state.get("ui_emitted", False):
                    final_content = getattr(output, "content", "") if output else ""
                    final_kwargs = getattr(output, "additional_kwargs", {}) if output else {}
                    rendered = _format_message_for_ui(final_content, final_kwargs)
                    if rendered:
                        if _phase_allows_nonstream_output(phase):
                            if phase in _VISIBLE_RESPONSE_PHASES:
                                pending_nonstream_output = rendered
                                fallback_nonstream_output = None
                                print(
                                    f"[RAG {trace_id}] candidate_answer | phase={phase} | "
                                    f"target=pending | chars={len(rendered)}"
                                )
                            else:
                                fallback_nonstream_output = rendered
                                print(
                                    f"[RAG {trace_id}] candidate_answer | phase={phase or 'unknown'} | "
                                    f"target=fallback | chars={len(rendered)}"
                                )
                        else:
                            print(
                                f"[RAG {trace_id}] candidate_discarded | phase={phase or 'unknown'} | "
                                "reason=decision_phase_non_user_visible"
                            )

        elif kind == "on_tool_end":
            # Capture sources from tool output
            data = event.get("data", {})
            output = data.get("output")
            tool_name = getattr(output, "name", "unknown_tool")
            output_len = len(getattr(output, "content", "") or "") if output else 0
            print(
                f"[RAG {trace_id}] tool_end | name={tool_name} | output_chars={output_len}"
            )

            metadata = extract_artifacts(data.get("artifact")) or extract_artifacts(output)
            print(f"[RAG {trace_id}] tool_artifacts | count={len(metadata)}")
            if metadata:
                source_preview = []
                for item in metadata:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") in {"plot", "final_response"}:
                        continue
                    title = str(item.get("title") or "").strip()
                    url = str(item.get("url") or "").strip()
                    if not title and not url:
                        continue
                    source_preview.append(f"{title} -> {url}")
                    if len(source_preview) >= 5:
                        break
                if source_preview:
                    print(f"[RAG {trace_id}] tool_sources | {' | '.join(source_preview)}")

            if metadata:
                final_response = find_final_response_artifact(metadata)
                if final_response is not None:
                    pending_nonstream_output = final_response
                    fallback_nonstream_output = None
                    print(
                        f"[RAG {trace_id}] tool_final_response | chars={len(final_response)}"
                    )

                for item in metadata:
                    if not isinstance(item, dict):
                        continue

                    item_type = item.get("type")
                    if item_type == "plot":
                        if item not in context["visual_artifacts"]:
                            context["visual_artifacts"].append(item)
                    elif item_type == "final_response":
                        continue
                    else:
                        if item not in context["sources"]:
                            context["sources"].append(item)
                            
    # Final step: append visual artifacts and source citations
    if not answer_emitted:
        if pending_nonstream_output:
            print(
                f"[RAG {trace_id}] emit_final | source=pending | chars={len(pending_nonstream_output)}"
            )
            yield pending_nonstream_output
            answer_emitted = True
        elif fallback_nonstream_output:
            print(
                f"[RAG {trace_id}] emit_final | source=fallback | chars={len(fallback_nonstream_output)}"
            )
            yield fallback_nonstream_output
            answer_emitted = True

    if last_reasoning_snapshot:
        context["last_turn_reasoning"] = last_reasoning_snapshot

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
    print(
        f"[RAG {trace_id}] complete | answer_emitted={answer_emitted} | "
        f"sources={len(context.get('sources', []))} | "
        f"visuals={len(context.get('visual_artifacts', []))} | "
        f"reasoning_chars={len(context.get('last_turn_reasoning', '') or '')}"
    )
