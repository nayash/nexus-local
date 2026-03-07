from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.agents.nodes import _detect_user_language, _should_use_reasoning_mode


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
