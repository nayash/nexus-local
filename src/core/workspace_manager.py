import datetime
import os
from typing import Dict, Optional, Tuple

from src.core.database import WatchedPathsRepository, WorkspaceRepository
from src.core.services.watcher import WatcherService
from src.rag.ingestion import ingest_path
from src.rag.ingestion_multimodal import (
    has_indexed_documents_for_prefix,
    reassign_workspace_prefix,
)


class WorkspaceManager:
    """
    Coordinates chat workspaces, initial ingestion, and background watching.

    Workspaces reuse the shared vector store and scope retrieval through
    workspace_id metadata rather than creating separate databases.
    """

    def __init__(self):
        self.workspace_repo = WorkspaceRepository()
        self.watched_repo = WatchedPathsRepository()
        self.watcher_service = WatcherService()

    def _workspace_name_from_path(self, path: str) -> str:
        name = os.path.basename(path.rstrip(os.sep))
        return name or path

    def _mark_workspace_ingested(self, workspace_id: str):
        now = datetime.datetime.now()
        self.workspace_repo.update_workspace_status(
            workspace_id,
            ingest_status="ready",
            last_ingested_at=now,
        )

    def _mark_workspace_watched(self, workspace_id: str):
        self.workspace_repo.update_workspace_status(
            workspace_id,
            last_watched_at=datetime.datetime.now(),
        )

    def _create_workspace(self, path: str) -> Dict:
        workspace_id = self.workspace_repo.create_workspace(
            name=self._workspace_name_from_path(path),
            root_path=path,
            ingest_status="pending",
        )
        return self.workspace_repo.get_workspace(workspace_id)

    def _ingest_workspace_backfill(self, path: str, workspace_id: str):
        self.workspace_repo.update_workspace_status(workspace_id, ingest_status="indexing")
        ingest_path(path, strategy="multimodal", workspace_id=workspace_id)
        self._mark_workspace_ingested(workspace_id)

    def _ensure_workspace_watch(self, path: str, workspace_id: str):
        watched = self.watched_repo.get_watched_path_by_path(path)
        if watched and watched.get("watch_mode") == "workspace_ingest" and bool(watched.get("recursive")):
            self.watcher_service.start()
            if path not in self.watcher_service.watches:
                self.watcher_service.watch_path(
                    path,
                    watch_mode="workspace_ingest",
                    workspace_id=workspace_id,
                    recursive=True,
                )
            self._mark_workspace_watched(workspace_id)
            return

        if watched:
            self.watcher_service.unwatch_path(path)
            self.watched_repo.remove_watched_path(watched["id"])

        self.watcher_service.start()
        self.watched_repo.add_watched_path(
            path,
            table_name=f"workspace:{workspace_id}",
            strategy="workspace_ingest",
            workspace_id=workspace_id,
            watch_mode="workspace_ingest",
            recursive=True,
        )
        self.watcher_service.watch_path(
            path,
            watch_mode="workspace_ingest",
            workspace_id=workspace_id,
            recursive=True,
        )
        self._mark_workspace_watched(workspace_id)

    def ensure_workspace(self, path: str) -> Tuple[bool, str, Optional[Dict]]:
        normalized_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(normalized_path):
            return False, "Workspace path does not exist.", None

        workspace = self.workspace_repo.get_workspace_by_path(normalized_path)
        if workspace is None:
            workspace = self._create_workspace(normalized_path)

        workspace_id = workspace["id"]
        watched = self.watched_repo.get_watched_path_by_path(normalized_path)

        if watched:
            if watched.get("workspace_id") != workspace_id:
                self.watched_repo.update_watched_path_workspace(watched["id"], workspace_id)

            has_scoped_docs = has_indexed_documents_for_prefix(normalized_path, workspace_id=workspace_id)
            has_any_docs = has_indexed_documents_for_prefix(normalized_path)

            if has_any_docs and not has_scoped_docs:
                reassign_workspace_prefix(normalized_path, workspace_id)
                self._mark_workspace_ingested(workspace_id)
                has_scoped_docs = True

            if not has_scoped_docs:
                self._ingest_workspace_backfill(normalized_path, workspace_id)

            self._ensure_workspace_watch(normalized_path, workspace_id)
            refreshed = self.workspace_repo.get_workspace(workspace_id)
            return True, "Workspace linked and synced from existing watched folder.", refreshed

        already_scoped = has_indexed_documents_for_prefix(normalized_path, workspace_id=workspace_id)
        if already_scoped:
            self._mark_workspace_ingested(workspace_id)
        elif has_indexed_documents_for_prefix(normalized_path):
            reassign_workspace_prefix(normalized_path, workspace_id)
            self._mark_workspace_ingested(workspace_id)
        else:
            self._ingest_workspace_backfill(normalized_path, workspace_id)

        self._ensure_workspace_watch(normalized_path, workspace_id)
        refreshed = self.workspace_repo.get_workspace(workspace_id)
        return True, "Workspace is ready for chat.", refreshed

    def get_workspace(self, workspace_id: str) -> Optional[Dict]:
        return self.workspace_repo.get_workspace(workspace_id)

    def list_workspaces(self):
        return self.workspace_repo.list_workspaces()
