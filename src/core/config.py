import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Resolve Project Root (2 levels up from src/core/config.py -> src/core -> src -> root)
    # Actually config.py is in src/core, so:
    # dirname(__file__) -> src/core
    # dirname(...) -> src
    # dirname(...) -> root
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, "..", ".."))
    
    SUPPORTED_MODELS = ["llama3.1", "mistral-nemo", "qwen3", "hermes3:8b"]
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
