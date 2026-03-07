from src.tools.tool_results import (
    build_final_response_artifact,
    extract_artifacts,
    extract_final_response,
)


class _FakeToolMessage:
    def __init__(self, artifact):
        self.artifact = artifact


def test_extract_final_response_from_tool_message_artifact():
    payload = _FakeToolMessage(
        artifact=[
            {"title": "Local File (txt): demo.txt", "url": "/tmp/demo.txt", "type": "local"},
            build_final_response_artifact("hello"),
        ]
    )
    assert extract_final_response(payload) == "hello"


def test_extract_artifacts_from_legacy_tuple():
    payload = ("ignored", [{"type": "local", "title": "x", "url": "/tmp/x"}])
    assert extract_artifacts(payload) == [{"type": "local", "title": "x", "url": "/tmp/x"}]
