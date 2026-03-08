from types import SimpleNamespace

from src.ui.agent_interface import (
    _consume_stream_chunk,
    _create_llm_buffer,
    _extract_response_phase,
    _format_message_for_ui,
    _is_reasoning_followup_query,
    _phase_allows_nonstream_output,
    _phase_allows_ui_stream,
    _strip_think_markup_for_ui,
)


def test_format_message_for_ui_keeps_content_only():
    rendered = _format_message_for_ui(
        "Final answer",
        {"reasoning_content": "hidden chain of thought"},
    )
    assert rendered == "Final answer"
    assert "<think>" not in rendered


def test_stream_chunk_emits_only_answer_content_and_tracks_reasoning():
    buffer_state = _create_llm_buffer()
    chunk = SimpleNamespace(
        content="Visible answer part",
        additional_kwargs={"reasoning_content": "internal reasoning part"},
    )

    emitted = _consume_stream_chunk(buffer_state, chunk)

    assert emitted == ["Visible answer part"]
    assert buffer_state["reasoning_parts"] == ["internal reasoning part"]
    assert buffer_state["content_parts"] == ["Visible answer part"]


def test_reasoning_followup_query_detection():
    assert _is_reasoning_followup_query("What was your reasoning behind the last answer?") is True
    assert _is_reasoning_followup_query("Explain the MemShuffle card game idea") is False


def test_strip_think_markup_for_ui_handles_unclosed_tag():
    raw = "Visible part\n<think>internal text never closed"
    assert _strip_think_markup_for_ui(raw) == "Visible part"


def test_format_message_for_ui_strips_think_blocks():
    rendered = _format_message_for_ui("A<think>x</think>B", {})
    assert rendered == "AB"


def test_extract_response_phase_from_metadata():
    event = {"metadata": {"nexus_phase": "final"}}
    assert _extract_response_phase(event) == "final"


def test_extract_response_phase_from_tags():
    event = {"metadata": {}, "tags": ["x", "nexus_phase:final_retry"]}
    assert _extract_response_phase(event) == "final_retry"


def test_phase_visibility_rules():
    assert _phase_allows_ui_stream("final") is True
    assert _phase_allows_ui_stream("decision") is False
    assert _phase_allows_nonstream_output("decision") is False
    assert _phase_allows_nonstream_output("unknown") is True
