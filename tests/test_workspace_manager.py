import os
import shutil
import tempfile
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.database import WatchedPathsRepository, WorkspaceRepository
from src.core.workspace_manager import WorkspaceManager


def test_workspace_manager_reuses_existing_watched_folder(monkeypatch):
    temp_dir = tempfile.mkdtemp(prefix="nexus_workspace_")
    workspace_repo = WorkspaceRepository()
    watched_repo = WatchedPathsRepository()

    try:
        watched_repo.add_watched_path(
            temp_dir,
            table_name="watch:test",
            strategy="organize_and_ingest",
            watch_mode="organize_and_ingest",
            recursive=False,
        )

        monkeypatch.setattr(
            "src.core.workspace_manager.has_indexed_documents_for_prefix",
            lambda path, workspace_id=None: workspace_id is None,
        )
        reassigned = []
        monkeypatch.setattr(
            "src.core.workspace_manager.reassign_workspace_prefix",
            lambda path, workspace_id: reassigned.append((path, workspace_id)),
        )
        monkeypatch.setattr(
            "src.core.workspace_manager.ingest_path",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ingest_path should not run")),
        )

        manager = WorkspaceManager()
        manager.watcher_service.start = lambda: None
        manager.watcher_service.unwatch_path = lambda *args, **kwargs: None
        manager.watcher_service.watch_path = lambda *args, **kwargs: None

        success, _, workspace = manager.ensure_workspace(temp_dir)

        assert success is True
        assert workspace is not None
        assert reassigned == [(os.path.abspath(temp_dir), workspace["id"])]

        watched_row = watched_repo.get_watched_path_by_path(os.path.abspath(temp_dir))
        assert watched_row is not None
        assert watched_row["workspace_id"] == workspace["id"]
        assert watched_row["watch_mode"] == "workspace_ingest"
        assert watched_row["recursive"] == 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        workspace = workspace_repo.get_workspace_by_path(os.path.abspath(temp_dir))
        if workspace:
            conn = workspace_repo._get_conn()
            conn.execute("DELETE FROM workspaces WHERE id = ?", (workspace["id"],))
            conn.commit()
            conn.close()
        watched = watched_repo.get_watched_path_by_path(os.path.abspath(temp_dir))
        if watched:
            watched_repo.remove_watched_path(watched["id"])


def test_workspace_manager_backfills_when_watched_folder_has_no_index(monkeypatch):
    temp_dir = tempfile.mkdtemp(prefix="nexus_workspace_reingest_")
    workspace_repo = WorkspaceRepository()
    watched_repo = WatchedPathsRepository()

    try:
        watched_repo.add_watched_path(
            temp_dir,
            table_name="watch:test",
            strategy="organize_and_ingest",
            watch_mode="organize_and_ingest",
            recursive=False,
        )

        monkeypatch.setattr(
            "src.core.workspace_manager.has_indexed_documents_for_prefix",
            lambda path, workspace_id=None: False,
        )
        monkeypatch.setattr(
            "src.core.workspace_manager.reassign_workspace_prefix",
            lambda path, workspace_id: (_ for _ in ()).throw(AssertionError("reassign should not run")),
        )
        ingested = []
        monkeypatch.setattr(
            "src.core.workspace_manager.ingest_path",
            lambda path, strategy="multimodal", workspace_id="global": ingested.append((path, workspace_id)),
        )

        manager = WorkspaceManager()
        manager.watcher_service.start = lambda: None
        manager.watcher_service.unwatch_path = lambda *args, **kwargs: None
        manager.watcher_service.watch_path = lambda *args, **kwargs: None

        success, _, workspace = manager.ensure_workspace(temp_dir)

        assert success is True
        assert workspace is not None
        assert ingested == [(os.path.abspath(temp_dir), workspace["id"])]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        workspace = workspace_repo.get_workspace_by_path(os.path.abspath(temp_dir))
        if workspace:
            conn = workspace_repo._get_conn()
            conn.execute("DELETE FROM workspaces WHERE id = ?", (workspace["id"],))
            conn.commit()
            conn.close()
        watched = watched_repo.get_watched_path_by_path(os.path.abspath(temp_dir))
        if watched:
            watched_repo.remove_watched_path(watched["id"])
