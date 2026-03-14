import os

from src.rag import ingestion as ingestion_module


def test_ingest_path_includes_ingestible_unknown_extensions(tmp_path, monkeypatch):
    folder = tmp_path / "abalone"
    folder.mkdir()
    (folder / "abalone.data").write_text("rings,weight\n1,0.5\n", encoding="utf-8")
    (folder / "abalone.names").write_text("Dataset description\n", encoding="utf-8")
    (folder / "ignore.bin").write_bytes(b"\x00\x01\x02\x03")

    ingested_paths = []

    def fake_ingest_file(file_path, progress_callback=None, workspace_id="global"):
        ingested_paths.append(file_path)
        return True, 1, "hash"

    monkeypatch.setattr(ingestion_module, "ingest_file_multimodal", fake_ingest_file)

    success, message, _ = ingestion_module.ingest_path(str(folder))

    assert success is True
    assert "Successfully ingested 2 files" in message
    names = sorted(os.path.basename(path) for path in ingested_paths)
    assert names == ["abalone.data", "abalone.names"]
