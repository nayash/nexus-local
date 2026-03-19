from src.rag import ingestion_multimodal as ingestion_module


def test_lexical_source_path_filter_stays_inside_workspace(monkeypatch):
    monkeypatch.setattr(
        ingestion_module,
        "load_rows",
        lambda table_name: [
            {
                "workspace_id": "workspace-a",
                "source_path": "/tmp/Index",
                "file_name": "Index",
                "title": "Index",
            },
            {
                "workspace_id": "global",
                "source_path": "/tmp/OutsideIndex",
                "file_name": "Index",
                "title": "Index outside",
            },
        ],
    )

    sql_filter = ingestion_module._lexical_source_path_filter(
        'explain content of the "Index" file',
        file_filter=None,
        workspace_id="workspace-a",
    )

    assert "workspace_id = 'workspace-a'" in sql_filter
    assert "/tmp/Index" in sql_filter
    assert "/tmp/OutsideIndex" not in sql_filter
