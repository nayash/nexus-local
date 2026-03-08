from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agents.nodes import (
    _clean_visible_response_text,
    _detect_user_language,
    _inject_forced_tool_call,
    _should_regenerate_final_answer,
    _should_force_local_search_fallback,
    _should_use_reasoning_mode,
)


def test_detect_user_language_defaults_to_english():
    assert _detect_user_language("Give me writing tips from my notes") == "English"


def test_detect_user_language_respects_explicit_request():
    assert _detect_user_language("Please reply in thai") == "Thai"


def test_reasoning_mode_not_forced_by_large_tool_payload_on_simple_query():
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="Give me writing tips from my notes"),
        ToolMessage(content=("x" * 5000), tool_call_id="tool-1"),
    ]
    assert _should_use_reasoning_mode(messages) is False


def test_force_local_fallback_for_personal_local_query():
    messages = [
        SystemMessage(content="system"),
        AIMessage(content="Earlier result: Local File (md): [Game] MemShuffle ..."),
        HumanMessage(content="Explain the MemShuffle card game idea I had"),
    ]
    assert _should_force_local_search_fallback(messages, "Explain the MemShuffle card game idea I had") is True


def test_force_local_fallback_not_triggered_for_explicit_web_query():
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="What is the latest news about AI today?"),
    ]
    assert _should_force_local_search_fallback(messages, "What is the latest news about AI today?") is False


def test_inject_forced_local_tool_call_when_model_skips_tools():
    messages = [
        SystemMessage(content="system"),
        AIMessage(content="Local File (md): [Game] MemShuffle ..."),
        HumanMessage(content="Explain the MemShuffle card game idea I had"),
    ]
    response = AIMessage(content="I cannot find that file.", tool_calls=[])

    injected = _inject_forced_tool_call(
        response,
        messages,
        "Explain the MemShuffle card game idea I had",
        focused_file=None,
    )

    assert injected is True
    assert response.tool_calls
    assert response.tool_calls[0]["name"] == "local_search_tool"
    assert response.tool_calls[0]["args"]["query"] == "Explain the MemShuffle card game idea I had"


def test_does_not_inject_forced_tool_call_on_tool_follow_up_turn():
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="Explain the MemShuffle card game idea I had"),
        ToolMessage(content="retrieved context", tool_call_id="tool-1"),
    ]
    response = AIMessage(content="intermediate answer", tool_calls=[])

    injected = _inject_forced_tool_call(
        response,
        messages,
        "Explain the MemShuffle card game idea I had",
        focused_file=None,
    )

    assert injected is False
    assert response.tool_calls == []


def test_should_not_regenerate_when_tool_followup_already_has_answer():
    response = AIMessage(content="Here is the final answer.", tool_calls=[])
    assert _should_regenerate_final_answer("tool", response) is False


def test_should_regenerate_when_human_turn_has_no_tool_call():
    response = AIMessage(content="Direct answer", tool_calls=[])
    assert _should_regenerate_final_answer("human", response) is True


def test_should_regenerate_when_tool_followup_is_missing_context_refusal():
    response = AIMessage(
        content="I don't have access to that file. Could you describe it?",
        tool_calls=[],
    )
    assert _should_regenerate_final_answer("tool", response) is True


def test_clean_visible_response_text_drops_unclosed_think_tail():
    content = "Final intro.\n<think>internal reasoning without close"
    cleaned, had_think = _clean_visible_response_text(content)
    assert had_think is True
    assert cleaned == "Final intro."


def test_clean_visible_response_text_removes_well_formed_think_block():
    content = "Before<think>secret</think>After"
    cleaned, had_think = _clean_visible_response_text(content)
    assert had_think is True
    assert cleaned == "BeforeAfter"
