from dataclasses import dataclass
from typing import Callable, Optional

from src.core.preflight import run_preflight
from src.rag.ingestion import init_knowledge


@dataclass
class StartupResult:
    success: bool
    web_search_enabled: bool
    error_message: Optional[str] = None
    feature_readiness: Optional[dict] = None


class StartupManager:
    """
    Runs non-mutating startup checks.
    Installation/provisioning now belongs to explicit bootstrap commands.
    """

    def __init__(self, is_dev: bool = False):
        self.is_dev = is_dev

    def run_checks(self, on_update: Callable[[str, float, bool], None]) -> StartupResult:
        try:
            if self.is_dev:
                return StartupResult(
                    success=True,
                    web_search_enabled=True,
                    feature_readiness={},
                )

            on_update("Preparing local data directory...", 0.1, False)
            report = run_preflight(
                install_ollama=False,
                start_ollama=False,
                pull_models=False,
                install_pyodide=False,
                build_docker_image=False,
                download_onnx=False,
                check_multimodal_embedder=False,
                migrate_legacy_data=True,
            )

            on_update("Running environment preflight...", 0.5, False)
            readiness = report.to_feature_readiness()

            if not report.core_ready:
                error_message = report.actionable_error()
                return StartupResult(
                    success=False,
                    web_search_enabled=report.web_search_enabled,
                    error_message=error_message,
                    feature_readiness=readiness,
                )

            on_update("Initializing knowledge base...", 0.9, False)
            init_knowledge()

            optional_failures = [
                f"{name}: {check.summary}"
                for name, check in report.checks.items()
                if check.optional and not check.ready
            ]
            if optional_failures:
                print("Optional feature checks not ready:")
                for failure in optional_failures:
                    print(f"- {failure}")

            on_update("Ready!", 1.0, False)
            return StartupResult(
                success=True,
                web_search_enabled=report.web_search_enabled,
                feature_readiness=readiness,
            )
        except Exception as exc:
            return StartupResult(
                success=False,
                web_search_enabled=False,
                error_message=f"Unexpected startup error: {exc}",
                feature_readiness={},
            )

