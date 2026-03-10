from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import requests

from src.core.config import Config


REQUIRED_OLLAMA_MODELS = ("llama3.1", "nomic-embed-text")

TEXT_MODEL_CANDIDATES = ("text_encoder.onnx", "text_model.onnx", "model_text.onnx")
VISION_MODEL_CANDIDATES = ("vision_encoder.onnx", "image_encoder.onnx", "vision_model.onnx", "model_vision.onnx")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)

DEFAULT_MODEL_URLS = {
    "text_model.onnx": "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model.onnx",
    "vision_model.onnx": "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model.onnx",
    "tokenizer.json": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer.json",
    "tokenizer_config.json": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer_config.json",
    "special_tokens_map.json": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/special_tokens_map.json",
    "vocab.json": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json",
    "merges.txt": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt",
}


@dataclass
class FeatureCheck:
    ready: bool
    summary: str
    action: str = ""
    details: str = ""
    optional: bool = False


@dataclass
class PreflightReport:
    checks: Dict[str, FeatureCheck]
    web_search_enabled: bool
    core_ready: bool
    notes: List[str]

    def to_feature_readiness(self) -> Dict[str, dict]:
        return {name: asdict(check) for name, check in self.checks.items()}

    def blocking_failures(self) -> List[Tuple[str, FeatureCheck]]:
        blockers = []
        for key in ("dependencies", "ollama", "ollama_service", "models"):
            check = self.checks.get(key)
            if check and not check.ready:
                blockers.append((key, check))
        return blockers

    def actionable_error(self) -> str:
        blockers = self.blocking_failures()
        if not blockers:
            return ""
        lines = ["Startup preflight failed:"]
        for key, check in blockers:
            lines.append(f"- {key}: {check.summary}")
            if check.action:
                lines.append(f"  action: {check.action}")
        return "\n".join(lines)


def _path_has_content(path: str) -> bool:
    if not os.path.exists(path):
        return False
    if os.path.isfile(path):
        return os.path.getsize(path) > 0
    return any(True for _ in os.scandir(path))


def ensure_data_layout(migrate_legacy: bool = True) -> List[str]:
    notes: List[str] = []
    directories = [
        Config.DATA_DIR,
        os.path.dirname(Config.SQLITE_PATH),
        Config.LANCEDB_PATH,
        Config.DOCSTORE_PATH,
        Config.MULTIMODAL_IMAGE_CACHE_DIR,
        Config.PYODIDE_NPM_DIR,
        Config.MULTIMODAL_MODEL_DIR,
    ]
    for path in directories:
        os.makedirs(path, exist_ok=True)

    if not migrate_legacy:
        return notes

    migration_sentinel = os.path.join(Config.DATA_DIR, ".legacy_migration_complete")
    if os.path.exists(migration_sentinel):
        return notes

    legacy_data_root = os.path.join(Config.PROJECT_ROOT, "data")
    migrations = [
        (os.path.join(legacy_data_root, "lancedb"), Config.LANCEDB_PATH),
        (os.path.join(legacy_data_root, "sqlite"), os.path.dirname(Config.SQLITE_PATH)),
        (os.path.join(legacy_data_root, "docstore"), Config.DOCSTORE_PATH),
        (os.path.join(legacy_data_root, "ingested_images"), Config.MULTIMODAL_IMAGE_CACHE_DIR),
    ]

    for src, dst in migrations:
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        if not os.path.isdir(src) or _path_has_content(dst):
            continue
        shutil.copytree(src, dst, dirs_exist_ok=True)
        notes.append(f"Migrated data from {src} to {dst}")

    legacy_settings = os.path.join(Config.PROJECT_ROOT, "user_settings.json")
    if (
        os.path.isfile(legacy_settings)
        and os.path.abspath(legacy_settings) != os.path.abspath(Config.USER_SETTINGS_PATH)
        and not os.path.exists(Config.USER_SETTINGS_PATH)
    ):
        os.makedirs(os.path.dirname(Config.USER_SETTINGS_PATH), exist_ok=True)
        shutil.copy2(legacy_settings, Config.USER_SETTINGS_PATH)
        notes.append(f"Migrated settings from {legacy_settings} to {Config.USER_SETTINGS_PATH}")

    legacy_models = os.path.join(Config.PROJECT_ROOT, "models", "clip_onnx")
    if (
        os.path.isdir(legacy_models)
        and os.path.abspath(legacy_models) != os.path.abspath(Config.MULTIMODAL_MODEL_DIR)
        and not _path_has_content(Config.MULTIMODAL_MODEL_DIR)
    ):
        shutil.copytree(legacy_models, Config.MULTIMODAL_MODEL_DIR, dirs_exist_ok=True)
        notes.append(f"Migrated multimodal models from {legacy_models} to {Config.MULTIMODAL_MODEL_DIR}")

    try:
        Path(migration_sentinel).write_text("ok\n")
    except OSError:
        pass
    return notes


def _check_python_dependencies() -> FeatureCheck:
    missing = []
    for module_name in (
        "flet",
        "langgraph",
        "lancedb",
        "pypdf",
        "pandas",
        "watchdog",
        "requests",
        "pydantic",
        "lark",
        "langchain_community",
        "langchain_ollama",
    ):
        try:
            __import__(module_name)
        except Exception:
            missing.append(module_name)

    if missing:
        joined = ", ".join(missing)
        return FeatureCheck(
            ready=False,
            summary=f"Missing Python dependencies: {joined}",
            action="Install core deps with `pip install .` (or `pip install -r requirements.txt`).",
        )

    return FeatureCheck(ready=True, summary="Core Python dependencies are available.")


def _check_web_search() -> Tuple[bool, FeatureCheck]:
    provider = (Config.WEB_SEARCH_PROVIDER or "").strip().lower()
    if provider == "tavily":
        enabled = bool(Config.TAVILY_API_KEY)
        return enabled, FeatureCheck(
            ready=enabled,
            summary="Tavily web search configured." if enabled else "TAVILY_API_KEY is missing.",
            action="" if enabled else "Set TAVILY_API_KEY in your environment to enable web search.",
            optional=True,
        )
    if provider == "serper":
        enabled = bool(Config.SERPER_API_KEY)
        return enabled, FeatureCheck(
            ready=enabled,
            summary="Serper web search configured." if enabled else "SERPER_API_KEY is missing.",
            action="" if enabled else "Set SERPER_API_KEY in your environment to enable web search.",
            optional=True,
        )
    return False, FeatureCheck(
        ready=False,
        summary=f"Unknown WEB_SEARCH_PROVIDER='{provider}'",
        action="Set WEB_SEARCH_PROVIDER to 'tavily' or 'serper'.",
        optional=True,
    )


def _install_ollama() -> Tuple[bool, str]:
    system = platform.system()
    try:
        if system == "Windows":
            url = "https://ollama.com/download/OllamaSetup.exe"
            setup_path = Path(os.getenv("TEMP", ".")) / "OllamaSetup.exe"
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            with open(setup_path, "wb") as handle:
                shutil.copyfileobj(response.raw, handle)
            subprocess.run([str(setup_path)], check=True)
            return True, "Ollama installer executed."

        if system in {"Linux", "Darwin"}:
            subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
            return True, "Ollama install script executed."
    except Exception as exc:
        return False, str(exc)

    return False, f"Unsupported OS for Ollama auto-install: {system}"


def _check_ollama_binary(auto_install: bool = False) -> FeatureCheck:
    if shutil.which("ollama"):
        return FeatureCheck(ready=True, summary="Ollama binary found.")

    if auto_install:
        installed, detail = _install_ollama()
        if installed and shutil.which("ollama"):
            return FeatureCheck(ready=True, summary="Ollama installed successfully.")
        return FeatureCheck(
            ready=False,
            summary="Failed to install Ollama automatically.",
            details=detail,
            action="Install Ollama manually from https://ollama.com/download and rerun setup.",
        )

    return FeatureCheck(
        ready=False,
        summary="Ollama binary not found in PATH.",
        action="Run `nexus-local setup --install-ollama` or install Ollama manually.",
    )


def _check_ollama_service(auto_start: bool = False) -> FeatureCheck:
    url = Config.OLLAMA_BASE_URL
    try:
        requests.get(url, timeout=1)
        return FeatureCheck(ready=True, summary="Ollama service is reachable.")
    except Exception:
        pass

    if auto_start:
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(10):
                time.sleep(1.5)
                try:
                    requests.get(url, timeout=1)
                    return FeatureCheck(ready=True, summary="Ollama service started.")
                except Exception:
                    continue
        except Exception as exc:
            return FeatureCheck(
                ready=False,
                summary="Failed to start Ollama service.",
                details=str(exc),
                action="Start Ollama manually with `ollama serve`.",
            )

    return FeatureCheck(
        ready=False,
        summary="Ollama service is not reachable.",
        action="Run `nexus-local setup --start-ollama` or start `ollama serve` manually.",
    )


def _check_ollama_models(auto_pull: bool = False) -> FeatureCheck:
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        installed = result.stdout
    except Exception:
        return FeatureCheck(
            ready=False,
            summary="Failed to list Ollama models.",
            action="Ensure Ollama service is running and accessible.",
        )

    missing = [
        model
        for model in REQUIRED_OLLAMA_MODELS
        if model not in installed and f"{model}:latest" not in installed
    ]
    if not missing:
        return FeatureCheck(ready=True, summary="Required Ollama models are present.")

    if auto_pull:
        failures = []
        for model in missing:
            try:
                subprocess.run(["ollama", "pull", model], check=True)
            except Exception as exc:
                failures.append(f"{model} ({exc})")
        if not failures:
            return FeatureCheck(ready=True, summary="Required Ollama models downloaded.")
        return FeatureCheck(
            ready=False,
            summary=f"Failed to pull models: {', '.join(failures)}",
            action="Run `ollama pull <model>` manually and rerun doctor.",
        )

    return FeatureCheck(
        ready=False,
        summary=f"Missing Ollama models: {', '.join(missing)}",
        action="Run `nexus-local setup --pull-models`.",
    )


def _node_env() -> dict:
    env = os.environ.copy()
    pyodide_node_modules = os.path.join(Config.PYODIDE_NPM_DIR, "node_modules")
    current = env.get("NODE_PATH", "")
    env["NODE_PATH"] = f"{pyodide_node_modules}{os.pathsep}{current}" if current else pyodide_node_modules
    return env


def _can_require_pyodide() -> bool:
    node = shutil.which("node")
    if not node:
        return False
    result = subprocess.run(
        [node, "-e", "require('pyodide'); process.exit(0);"],
        capture_output=True,
        text=True,
        env=_node_env(),
    )
    return result.returncode == 0


def _check_pyodide_runtime(auto_install: bool = False) -> FeatureCheck:
    node = shutil.which("node")
    if not node:
        return FeatureCheck(
            ready=False,
            summary="Node.js is not installed.",
            action="Install Node.js >= 18 to enable Pyodide sandbox.",
            optional=True,
        )

    if _can_require_pyodide():
        return FeatureCheck(ready=True, summary="Pyodide runtime is ready.", optional=True)

    npm = shutil.which("npm")
    if auto_install and npm:
        try:
            os.makedirs(Config.PYODIDE_NPM_DIR, exist_ok=True)
            subprocess.run(["npm", "install", "--prefix", Config.PYODIDE_NPM_DIR, "pyodide"], check=True)
        except Exception as exc:
            return FeatureCheck(
                ready=False,
                summary=f"Failed to install pyodide npm package: {exc}",
                action="Run `npm install --prefix <dir> pyodide` manually.",
                optional=True,
            )
        if _can_require_pyodide():
            return FeatureCheck(ready=True, summary="Pyodide runtime installed.", optional=True)

    return FeatureCheck(
        ready=False,
        summary="Pyodide npm package is missing.",
        action="Run `nexus-local setup --install-pyodide`.",
        optional=True,
    )


def _check_docker_sandbox(auto_build: bool = False) -> FeatureCheck:
    try:
        import docker
    except Exception:
        return FeatureCheck(
            ready=False,
            summary="Python docker package is missing.",
            action="Install with `pip install .[sandbox-docker]`.",
            optional=True,
        )

    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:
        return FeatureCheck(
            ready=False,
            summary=f"Docker daemon is unavailable: {exc}",
            action="Start Docker daemon to use docker sandbox.",
            optional=True,
        )

    try:
        client.images.get(Config.DOCKER_SANDBOX_IMAGE)
        return FeatureCheck(ready=True, summary="Docker sandbox image is ready.", optional=True)
    except Exception:
        pass

    if auto_build:
        dockerfile_path = Config.DOCKER_SANDBOX_DOCKERFILE
        if not os.path.isfile(dockerfile_path):
            return FeatureCheck(
                ready=False,
                summary=f"Dockerfile.sandbox not found at {dockerfile_path}",
                action="Set CODE_SANDBOX_ENGINE=pyodide or provide Dockerfile.sandbox.",
                optional=True,
            )
        try:
            dockerfile_rel = os.path.relpath(dockerfile_path, Config.DOCKER_SANDBOX_CONTEXT_DIR)
            client.images.build(
                path=Config.DOCKER_SANDBOX_CONTEXT_DIR,
                dockerfile=dockerfile_rel,
                tag=Config.DOCKER_SANDBOX_IMAGE,
                rm=True,
            )
            return FeatureCheck(ready=True, summary="Docker sandbox image built.", optional=True)
        except Exception as exc:
            return FeatureCheck(
                ready=False,
                summary=f"Failed to build docker sandbox image: {exc}",
                action="Build image manually with Docker.",
                optional=True,
            )

    return FeatureCheck(
        ready=False,
        summary=f"Docker image '{Config.DOCKER_SANDBOX_IMAGE}' is missing.",
        action="Run `nexus-local setup --build-docker-image`.",
        optional=True,
    )


def _download_multimodal_assets() -> Tuple[bool, str]:
    os.makedirs(Config.MULTIMODAL_MODEL_DIR, exist_ok=True)
    for filename, url in DEFAULT_MODEL_URLS.items():
        target = os.path.join(Config.MULTIMODAL_MODEL_DIR, filename)
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            with open(target, "wb") as handle:
                handle.write(response.content)
        except Exception as exc:
            return False, f"Failed to download {filename}: {exc}"
    return True, "Downloaded ONNX/tokenizer assets."


def _multimodal_asset_gaps() -> List[str]:
    gaps: List[str] = []
    model_dir = Config.MULTIMODAL_MODEL_DIR

    if not any(os.path.isfile(os.path.join(model_dir, name)) for name in TEXT_MODEL_CANDIDATES):
        gaps.append("text model (.onnx)")
    if not any(os.path.isfile(os.path.join(model_dir, name)) for name in VISION_MODEL_CANDIDATES):
        gaps.append("vision model (.onnx)")
    for filename in TOKENIZER_FILES:
        if not os.path.isfile(os.path.join(model_dir, filename)):
            gaps.append(filename)
    return gaps


def _check_multimodal_stack(auto_download_assets: bool = False, verify_embedder: bool = False) -> FeatureCheck:
    missing_modules = []
    for module_name in ("onnxruntime", "transformers", "tokenizers", "PIL", "fitz", "docx", "bs4"):
        try:
            __import__(module_name)
        except Exception:
            missing_modules.append(module_name)

    if missing_modules:
        return FeatureCheck(
            ready=False,
            summary=f"Missing multimodal Python deps: {', '.join(missing_modules)}",
            action="Install with `pip install .[multimodal]`.",
            optional=True,
        )

    gaps = _multimodal_asset_gaps()
    if gaps and auto_download_assets:
        downloaded, detail = _download_multimodal_assets()
        if not downloaded:
            return FeatureCheck(
                ready=False,
                summary="Failed to download multimodal assets.",
                details=detail,
                action="Download assets manually and rerun doctor.",
                optional=True,
            )
        gaps = _multimodal_asset_gaps()

    if gaps:
        return FeatureCheck(
            ready=False,
            summary=f"Missing multimodal assets: {', '.join(gaps)}",
            action="Run `nexus-local setup --download-onnx`.",
            optional=True,
        )

    provider_summary = "unknown"
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        provider_summary = ", ".join(providers)
    except Exception:
        provider_summary = "onnxruntime provider probe failed"

    if verify_embedder:
        try:
            from src.embeddings.multimodal_onnx import get_multimodal_embedder

            embedder = get_multimodal_embedder(force_refresh=True)
            if embedder is None:
                return FeatureCheck(
                    ready=False,
                    summary="Multimodal embedder failed to initialize.",
                    action="Run `nexus-local doctor` and inspect ONNX runtime output.",
                    optional=True,
                )
        except Exception as exc:
            return FeatureCheck(
                ready=False,
                summary=f"Multimodal embedder error: {exc}",
                action="Check model files and runtime providers.",
                optional=True,
            )

    return FeatureCheck(
        ready=True,
        summary=f"Multimodal stack is ready (providers: {provider_summary}).",
        optional=True,
    )


def run_preflight(
    *,
    install_ollama: bool = False,
    start_ollama: bool = False,
    pull_models: bool = False,
    install_pyodide: bool = False,
    build_docker_image: bool = False,
    download_onnx: bool = False,
    check_multimodal_embedder: bool = False,
    migrate_legacy_data: bool = True,
) -> PreflightReport:
    try:
        notes = ensure_data_layout(migrate_legacy=migrate_legacy_data)
    except Exception as exc:
        checks = {
            "dependencies": FeatureCheck(
                ready=False,
                summary=f"Failed to prepare local data directory: {exc}",
                action="Set NEXUS_DATA_DIR to a writable path and rerun setup.",
            )
        }
        return PreflightReport(
            checks=checks,
            web_search_enabled=False,
            core_ready=False,
            notes=[],
        )
    checks: Dict[str, FeatureCheck] = {}

    checks["dependencies"] = _check_python_dependencies()
    web_enabled, web_check = _check_web_search()
    checks["web_search"] = web_check

    checks["ollama"] = _check_ollama_binary(auto_install=install_ollama)
    if checks["ollama"].ready:
        checks["ollama_service"] = _check_ollama_service(auto_start=start_ollama)
    else:
        checks["ollama_service"] = FeatureCheck(
            ready=False,
            summary="Skipped: Ollama binary missing.",
            action="Install Ollama first.",
        )

    if checks["ollama_service"].ready:
        checks["models"] = _check_ollama_models(auto_pull=pull_models)
    else:
        checks["models"] = FeatureCheck(
            ready=False,
            summary="Skipped: Ollama service unavailable.",
            action="Start Ollama service first.",
        )

    checks["sandbox_pyodide"] = _check_pyodide_runtime(auto_install=install_pyodide)
    checks["sandbox_docker"] = _check_docker_sandbox(auto_build=build_docker_image)
    checks["multimodal"] = _check_multimodal_stack(
        auto_download_assets=download_onnx,
        verify_embedder=check_multimodal_embedder,
    )

    core_ready = (
        checks["dependencies"].ready
        and checks["ollama"].ready
        and checks["ollama_service"].ready
        and checks["models"].ready
    )

    return PreflightReport(
        checks=checks,
        web_search_enabled=web_enabled,
        core_ready=core_ready,
        notes=notes,
    )


def format_report(report: PreflightReport) -> str:
    lines: List[str] = []
    lines.append(f"Data directory: {Config.DATA_DIR}")
    for note in report.notes:
        lines.append(f"- note: {note}")
    lines.append(f"Core ready: {'yes' if report.core_ready else 'no'}")
    lines.append(f"Web search enabled: {'yes' if report.web_search_enabled else 'no'}")
    lines.append("")
    lines.append("Checks:")
    for name, check in report.checks.items():
        status = "OK" if check.ready else "FAIL"
        lines.append(f"[{status}] {name}: {check.summary}")
        if check.action:
            lines.append(f"  action: {check.action}")
        if check.details:
            lines.append(f"  details: {check.details}")
    return "\n".join(lines)
