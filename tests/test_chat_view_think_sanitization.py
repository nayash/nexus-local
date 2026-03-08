from src.ui.components.chat_view import ChatView


def test_strip_think_content_removes_well_formed_blocks():
    text = "Answer start <think>hidden reasoning</think> answer end"
    assert ChatView._strip_think_content(text) == "Answer start  answer end"


def test_strip_think_content_drops_unclosed_think_tail():
    text = "Visible answer.\n<think>internal reasoning"
    assert ChatView._strip_think_content(text) == "Visible answer."


def test_strip_think_content_removes_stray_closing_tag_only():
    text = "Hello</think> world"
    assert ChatView._strip_think_content(text) == "Hello world"
