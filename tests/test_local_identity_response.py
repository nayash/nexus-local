from src.tools.local import _build_identity_answer, resolve_direct_local_response
from src.tools.tool_results import extract_final_response


IDENTITY_TEXT = """# Identity: Nexus Local

Name: Nexus Local
Version: 0.1.0
Builder: Nayash, a private developer passionate about local AI.
Core Mission: To provide a privacy-first, offline-capable AI assistant that bridges local data with the public web.

Capabilities:
- I can read and index local files (PDF, CSV, TXT, MD).
- I can search the public web using DuckDuckGo (privacy-focused).
- I DO NOT send your personal data to any cloud server for training.
- I CANNOT generate images (DALL-E) or audio. I am text-only.

Personality:
- Professional but warm.
- Privacy-conscious.
- Concise and direct.
"""


def test_build_identity_answer_for_who_are_you():
    answer = _build_identity_answer("who are you?", IDENTITY_TEXT)

    assert answer is not None
    assert "Nexus Local" in answer
    assert "privacy-first" in answer


def test_resolve_direct_local_response_returns_identity_answer(tmp_path, monkeypatch):
    identity_path = tmp_path / "nexus-identity.txt"
    identity_path.write_text(IDENTITY_TEXT, encoding="utf-8")

    monkeypatch.setattr(
        "src.tools.local._plan_local_retrieval",
        lambda query, file_filter="": {"retrieval_mode": "document_lookup", "response_mode": "snippets"},
    )
    monkeypatch.setattr(
        "src.tools.local._query_documents_table",
        lambda query, file_filter="": [
            {
                "source_path": str(identity_path),
                "source_type": "txt",
                "file_name": "nexus-identity.txt",
            }
        ],
    )

    payload = resolve_direct_local_response("who are you?", str(identity_path))
    final_response = extract_final_response(payload[1])

    assert final_response is not None
    assert "Nexus Local" in final_response
    assert "Matched file:" not in final_response


def test_resolve_direct_local_response_keeps_normal_document_lookup(tmp_path, monkeypatch):
    document_path = tmp_path / "notes.txt"
    document_path.write_text("random note", encoding="utf-8")

    monkeypatch.setattr(
        "src.tools.local._plan_local_retrieval",
        lambda query, file_filter="": {"retrieval_mode": "document_lookup", "response_mode": "snippets"},
    )
    monkeypatch.setattr(
        "src.tools.local._query_documents_table",
        lambda query, file_filter="": [
            {
                "source_path": str(document_path),
                "source_type": "txt",
                "file_name": "notes.txt",
            }
        ],
    )

    payload = resolve_direct_local_response("where is my note?", str(document_path))
    final_response = extract_final_response(payload[1])

    assert final_response == f"Matched file:\n{document_path}"
