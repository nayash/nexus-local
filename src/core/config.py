import os
from dotenv import load_dotenv

load_dotenv()


def _env_flag(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

class Config:
    # Resolve Project Root (2 levels up from src/core/config.py -> src/core -> src -> root)
    # Actually config.py is in src/core, so:
    # dirname(__file__) -> src/core
    # dirname(...) -> src
    # dirname(...) -> root
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, "..", ".."))
    
    SUPPORTED_MODELS = ["llama3.1:8b", "mistral-nemo:12b", "qwen2.5:7b", "qwen3:8b", "qwen3.5:9b", "hermes3:8b", "gemma2:9b"]
    MAX_RESULTS = 5
    TIMEOUT = 10
    SEARXNG_BASE_URL = "http://localhost:8080"
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama3.1"
    LANCEDB_PATH = os.path.join(PROJECT_ROOT, "data", "lancedb")
    SQLITE_PATH = os.path.join(PROJECT_ROOT, "data", "sqlite", "nexus.db")

    # Web Search — switch provider via WEB_SEARCH_PROVIDER env var ("tavily" | "serper")
    WEB_SEARCH_PROVIDER: str = os.getenv("WEB_SEARCH_PROVIDER", "tavily")
    TAVILY_API_KEY: str  = os.getenv("TAVILY_API_KEY", "")
    SERPER_API_KEY: str  = os.getenv("SERPER_API_KEY", "")

    # Code Execution Sandbox — "docker" | "pyodide" (default: pyodide, no Docker required)
    CODE_SANDBOX_ENGINE: str = os.getenv("CODE_SANDBOX_ENGINE", "pyodide")
    DOCKER_SANDBOX_IMAGE: str = "nexus-sandbox:latest"
    DOCKER_TIMEOUT: int = 30
    DOCKER_MEM_LIMIT: str = "256m"
    PYODIDE_TIMEOUT: int = 30

    # Multimodal embeddings / RAG
    MULTIMODAL_EMBEDDINGS_ENABLED: bool = _env_flag("MULTIMODAL_EMBEDDINGS_ENABLED", "true")
    MULTIMODAL_MODEL_DIR: str = os.getenv(
        "MULTIMODAL_EMBED_MODEL_DIR",
        os.path.join(PROJECT_ROOT, "models", "clip_onnx"),
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
        os.path.join(PROJECT_ROOT, "data", "ingested_images"),
    )
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cuda")
    ORT_PROVIDER: str = os.getenv("ORT_PROVIDER", "CUDAExecutionProvider")
    ORT_PROVIDERS = [ORT_PROVIDER, "CPUExecutionProvider"] if ORT_PROVIDER else ["CPUExecutionProvider"]

    # RAG pipeline rollout
    RAG_PIPELINE_VERSION: str = os.getenv("RAG_PIPELINE_VERSION", "legacy").strip().lower()
    RAG_V2_SHADOW: bool = _env_flag("RAG_V2_SHADOW", "false")
