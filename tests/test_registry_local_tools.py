from src.tools.registry import (
    get_nexus_identity_tool,
    local_search_tool,
    lookup_local_files_tool,
)
from src.tools.tool_results import extract_final_response


def test_local_search_tool_stays_semantic_only(monkeypatch):
    class FakeResult:
        title = "Local File (md): writing_ideas.md"
        url = "/tmp/writing_ideas.md"

        def to_context_string(self):
            return "Writing ideas from local notes."

    monkeypatch.setattr(
        "src.tools.registry.search_local",
        lambda query, file_filter=None, workspace_id="": [FakeResult()],
    )

    content, metadata = local_search_tool.func("List down the writing ideas I had")

    assert "Writing ideas from local notes." in content
    assert metadata == [{"title": "Local File (md): writing_ideas.md", "url": "/tmp/writing_ideas.md", "type": "local"}]


def test_lookup_local_files_tool_uses_document_lookup_path(monkeypatch):
    monkeypatch.setattr(
        "src.tools.registry.resolve_direct_local_response",
        lambda query, file_filter="", workspace_id="": (
            "",
            [
                {"title": "Local File (md): project_report.md", "url": "/tmp/project_report.md", "type": "local"},
                {"type": "final_response", "content": "Matched file:\n/tmp/project_report.md"},
            ],
        ),
    )

    payload = lookup_local_files_tool.func("Which file is the project report?")
    assert extract_final_response(payload[1]) == "Matched file:\n/tmp/project_report.md"


def test_get_nexus_identity_tool_returns_identity_payload(monkeypatch):
    monkeypatch.setattr(
        "src.tools.registry.get_nexus_identity_response",
        lambda query: (
            "",
            [
                {"title": "Local File (txt): nexus-identity.txt", "url": "/tmp/nexus-identity.txt", "type": "local"},
                {"type": "final_response", "content": "I'm Nexus Local."},
            ],
        ),
    )

    payload = get_nexus_identity_tool.func("who are you?")
    assert extract_final_response(payload[1]) == "I'm Nexus Local."
