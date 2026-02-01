class Config:
    SUPPORTED_MODELS = ["llama3.1", "mistral-nemo", "qwen3"]
    MAX_RESULTS = 5
    TIMEOUT = 10
    SEARXNG_BASE_URL = "http://localhost:8080"
    OLLAMA_BASE_URL = "http://localhost:11434"
    LANCEDB_PATH = "data/lancedb"
    SQLITE_PATH = "data/sqlite/nexus.db"
