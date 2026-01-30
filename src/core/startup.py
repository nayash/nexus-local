import os
import sys
import shutil
import subprocess
import time
import threading
import requests
import platform
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class StartupResult:
    success: bool
    web_search_enabled: bool
    error_message: Optional[str] = None

class StartupManager:
    """
    Manages application startup checks and initialization.
    Ensures dependencies, connectivity, and background services (Ollama) are ready.
    """
    
    REQUIRED_MODELS = [
        "llama3.1",
        "nomic-embed-text"
    ]

    def run_checks(self, on_update: Callable[[str, float, bool], None]) -> StartupResult:
        """
        Runs all startup checks sequentially.
        
        Args:
            on_update: Callback for UI updates (message, progress, is_error).
        
        Returns:
            StartupResult: Result of the startup process.
        """
        try:
            # 1. Dependency Verification
            on_update("Verifying dependencies...", 0.1, False)
            if not self._check_dependencies():
                return StartupResult(False, False, "Corrupt installation. Missing dependencies.")
            
            # 2. Internet & Search Check
            on_update("Checking connectivity...", 0.2, False)
            web_search_enabled = self._check_internet_connection()
            if not web_search_enabled:
                 on_update("Offline Mode: Web search disabled", 0.25, False)

            # 3. Ollama Binary Check
            on_update("Checking AI Engine...", 0.3, False)
            if not self._check_ollama_binary(on_update):
                 return StartupResult(False, web_search_enabled, "Failed to install or find Ollama.")

            # 4. Ollama Service Check
            on_update("Connecting to AI Service...", 0.5, False)
            if not self._check_ollama_service(on_update):
                return StartupResult(False, web_search_enabled, "Failed to start AI Service.")

            # 5. AI Model Check
            on_update("Checking AI Models...", 0.7, False)
            if not self._check_models(on_update):
                return StartupResult(False, web_search_enabled, "Failed to download required models.")

            # 6. File System Check
            on_update("Verifying storage...", 0.9, False)
            if not self._check_filesystem():
                 return StartupResult(False, web_search_enabled, "Failed to initialize storage.")

            on_update("Ready!", 1.0, False)
            return StartupResult(True, web_search_enabled, None)

        except Exception as e:
            return StartupResult(False, False, f"Unexpected error during startup: {str(e)}")

    def _check_dependencies(self) -> bool:
        """Verify critical libraries are importable."""
        required_deps = [
            "flet", "langgraph", "duckduckgo_search", 
            "lancedb", "pypdf", "pandas"
        ]
        try:
            # We just try functionality or just existence? 
            # User said: "Verify that critical libraries are importable"
            # Since we are running inside the env, simple imports should work.
            # However, doing 'import flet' here just tests current env.
            import flet
            import langgraph
            import duckduckgo_search
            import lancedb
            import pypdf
            import pandas
            return True
        except ImportError:
            return False

    def _check_internet_connection(self) -> bool:
        """Ping DuckDuckGo to verify connectivity."""
        try:
            response = requests.get("https://duckduckgo.com", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _check_ollama_binary(self, on_update: Callable[[str, float, bool], None]) -> bool:
        """Check if ollama is in PATH, install if missing."""
        if shutil.which("ollama"):
            return True

        on_update("Downloading AI Engine...", 0.35, False)
        system = platform.system()
        
        try:
            if system == "Windows":
                # Download setup
                url = "https://ollama.com/download/OllamaSetup.exe"
                setup_path = Path(os.getenv("TEMP", ".")) / "OllamaSetup.exe"
                response = requests.get(url, stream=True)
                with open(setup_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
                
                on_update("Installing AI Engine (Please accept the prompt)...", 0.4, False)
                subprocess.run([str(setup_path)], check=True)
                
            elif system in ["Linux", "Darwin"]:
                 on_update("Installing AI Engine...", 0.4, False)
                 # piped curl execution is dangerous but requested by user specs
                 install_cmd = "curl -fsSL https://ollama.com/install.sh | sh"
                 subprocess.run(install_cmd, shell=True, check=True)
            else:
                 on_update(f"Unsupported OS: {system}", 1.0, True)
                 return False
            
            # Verify after install
            return shutil.which("ollama") is not None
            
        except subprocess.CalledProcessError:
            return False

    def _check_ollama_service(self, on_update: Callable[[str, float, bool], None]) -> bool:
        """Check if local API is reachable, start if not."""
        url = "http://localhost:11434"
        
        # Check if running
        try:
            requests.get(url, timeout=1)
            return True
        except requests.RequestException:
            pass # Not running

        # Attempt to start
        on_update("Starting AI Engine...", 0.55, False)
        try:
            # Start in background
            subprocess.Popen(["ollama", "serve"], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
            
            # Wait up to 15s
            for _ in range(7): # 7 * 2s = 14s + 1s initial check approx
                time.sleep(2)
                try:
                    requests.get(url, timeout=1)
                    return True
                except requests.RequestException:
                    continue
            
            return False
        except Exception:
            return False

    def _check_models(self, on_update: Callable[[str, float, bool], None]) -> bool:
        """Ensure required models are present."""
        try:
            # List models
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
            installed_models = result.stdout
            
            count = 0
            total = len(self.REQUIRED_MODELS)
            
            for model in self.REQUIRED_MODELS:
                if model not in installed_models and f"{model}:latest" not in installed_models:
                    on_update(f"Downloading model: {model}...", 0.7 + (0.2 * count/total), False)
                    if not self._pull_model(model, on_update):
                        return False
                count += 1
            return True
        except subprocess.CalledProcessError:
            return False

    def _pull_model(self, model_name: str, on_update: Callable[[str, float, bool], None]) -> bool:
        """
        Pull a model and parse output for progress.
        Required: Read stdout line-by-line to extract percentage.
        """
        try:
            process = subprocess.Popen(
                ["ollama", "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,            # Line buffered
                universal_newlines=True
            )
            
            if process.stdout is None:
                return False

            for line in process.stdout:
                # Example output: "pulling manifest" or "downloading ... 10%"
                # We want to send updates.
                line = line.strip()
                if not line:
                    continue
                    
                # Basic progress parsing (heuristic)
                # Ollama output format varies, but often contains "XX%"
                if "%" in line:
                     on_update(f"Downloading {model_name}: {line}", 0.8, False) # Keep float roughly static or calc
                
            process.wait()
            return process.returncode == 0
        except Exception:
            return False

    def _check_filesystem(self) -> bool:
        """Ensure data directories exist."""
        try:
            Path("./data/lancedb").mkdir(parents=True, exist_ok=True)
            Path("./data/sqlite").mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False
