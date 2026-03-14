import os
import platform
from dotenv import load_dotenv

# Suppress optional Hugging Face advisory warnings when torch is intentionally absent.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

load_dotenv()


def _env_flag(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _can_write_to_parent(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe_file = os.path.join(path, ".nexus_write_probe")
        with open(probe_file, "w", encoding="utf-8") as handle:
            handle.write("ok\n")
        os.remove(probe_file)
        return True
    except OSError:
        return False


class Config:
    # Resolve Project Root (2 levels up from src/core/config.py -> src/core -> src -> root)
    # Actually config.py is in src/core, so:
    # dirname(__file__) -> src/core
    # dirname(...) -> src
    # dirname(...) -> root
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, "..", ".."))

    @staticmethod
    def _default_data_dir() -> str:
        override = os.getenv("NEXUS_DATA_DIR", "").strip()
        if override:
            return os.path.abspath(os.path.expanduser(override))

        system = platform.system()
        if system == "Windows":
            appdata = os.getenv("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
            candidate = os.path.join(appdata, "Nexus Local")
            if _can_write_to_parent(candidate):
                return candidate
        if system == "Darwin":
            candidate = os.path.join(os.path.expanduser("~"), "Library", "Application Support", "Nexus Local")
            if _can_write_to_parent(candidate):
                return candidate
        else:
            xdg_data_home = os.getenv("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
            candidate = os.path.join(xdg_data_home, "nexus-local")
            if _can_write_to_parent(candidate):
                return candidate

        # Sandbox-safe fallback for restricted environments.
        project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        return os.path.join(project_root, "data")

    DATA_DIR = _default_data_dir.__func__()
    
    SUPPORTED_MODELS = ["llama3.1:8b", "mistral-nemo:12b", "qwen2.5:7b", "qwen3:8b", "qwen3.5:9b", "hermes3:8b", "gemma2:9b"]
    MAX_RESULTS = 5
    TIMEOUT = int(os.getenv("NEXUS_TIMEOUT", "30"))
    MANAGER_INTENT_TIMEOUT = int(os.getenv("MANAGER_INTENT_TIMEOUT", "30"))
    SEARXNG_BASE_URL = "http://localhost:8080"
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama3.1"
    LANCEDB_PATH = os.path.join(DATA_DIR, "lancedb")
    SQLITE_PATH = os.path.join(DATA_DIR, "sqlite", "nexus.db")
    DOCSTORE_PATH = os.path.join(DATA_DIR, "docstore")
    USER_SETTINGS_PATH = os.path.join(DATA_DIR, "user_settings.json")
    PYODIDE_NPM_DIR = os.path.join(DATA_DIR, "pyodide_npm")

    # Web Search — switch provider via WEB_SEARCH_PROVIDER env var ("tavily" | "serper")
    WEB_SEARCH_PROVIDER: str = os.getenv("WEB_SEARCH_PROVIDER", "tavily")
    TAVILY_API_KEY: str  = os.getenv("TAVILY_API_KEY", "")
    SERPER_API_KEY: str  = os.getenv("SERPER_API_KEY", "")

    # Code Execution Sandbox — "docker" | "pyodide" (default: pyodide, no Docker required)
    CODE_SANDBOX_ENGINE: str = os.getenv("CODE_SANDBOX_ENGINE", "pyodide")
    DOCKER_SANDBOX_IMAGE: str = "nexus-sandbox:latest"
    DOCKER_TIMEOUT: int = 30
    DOCKER_MEM_LIMIT: str = "256m"
    DOCKER_SANDBOX_CONTEXT_DIR: str = os.getenv("DOCKER_SANDBOX_CONTEXT_DIR", PROJECT_ROOT)
    DOCKER_SANDBOX_DOCKERFILE: str = os.getenv(
        "DOCKER_SANDBOX_DOCKERFILE",
        os.path.join(PROJECT_ROOT, "Dockerfile.sandbox"),
    )
    PYODIDE_TIMEOUT: int = 30

    # Multimodal embeddings / RAG
    MULTIMODAL_EMBEDDINGS_ENABLED: bool = _env_flag("MULTIMODAL_EMBEDDINGS_ENABLED", "true")
    MULTIMODAL_MODEL_DIR: str = os.getenv(
        "MULTIMODAL_EMBED_MODEL_DIR",
        os.path.join(DATA_DIR, "models", "clip_onnx"),
    )
    MULTIMODAL_PARENT_TABLE: str = os.getenv("MULTIMODAL_PARENT_TABLE", "nexus_parents")
    MULTIMODAL_TEXT_CHILD_TABLE: str = os.getenv(
        "MULTIMODAL_TEXT_CHILD_TABLE",
        "nexus_child_text_nomic",
    )
    MULTIMODAL_CLIP_CHILD_TABLE: str = os.getenv(
        "MULTIMODAL_CLIP_CHILD_TABLE",
        "nexus_child_clip",
    )
    MULTIMODAL_DOCUMENTS_TABLE: str = os.getenv("MULTIMODAL_DOCUMENTS_TABLE", "nexus_documents")
    MULTIMODAL_IMAGE_CACHE_DIR: str = os.getenv(
        "MULTIMODAL_IMAGE_CACHE_DIR",
        os.path.join(DATA_DIR, "ingested_images"),
    )
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cuda")
    ORT_PROVIDER: str = os.getenv("ORT_PROVIDER", "CUDAExecutionProvider")
    ORT_PROVIDERS = [ORT_PROVIDER, "CPUExecutionProvider"] if ORT_PROVIDER else ["CPUExecutionProvider"]

    # RAG pipeline rollout
    RAG_PIPELINE_VERSION: str = os.getenv("RAG_PIPELINE_VERSION", "legacy").strip().lower()
    RAG_V2_SHADOW: bool = _env_flag("RAG_V2_SHADOW", "false")
