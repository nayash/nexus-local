"""
Public entry point for sandboxed Python code execution.

Uses the Strategy Pattern — the active engine is controlled by
``Config.CODE_SANDBOX_ENGINE`` (``"docker"`` | ``"pyodide"``).
"""

import logging
from src.core.config import Config
from src.tools.sandbox.base import BaseSandboxExecutor, CodeExecutionResult

logger = logging.getLogger(__name__)

# Module-level singleton (created on first call)
_executor: BaseSandboxExecutor | None = None


def get_executor() -> BaseSandboxExecutor:
    """Factory — return the executor matching the configured engine."""
    global _executor

    if _executor is not None:
        return _executor

    engine = Config.CODE_SANDBOX_ENGINE.lower()

    if engine == "docker":
        from src.tools.sandbox.docker_executor import DockerSandboxExecutor
        _executor = DockerSandboxExecutor(
            image=Config.DOCKER_SANDBOX_IMAGE,
            timeout=Config.DOCKER_TIMEOUT,
            mem_limit=Config.DOCKER_MEM_LIMIT,
        )
        logger.info("🐳 Code execution engine: Docker")

    elif engine == "pyodide":
        from src.tools.sandbox.pyodide_executor import PyodideSandboxExecutor
        _executor = PyodideSandboxExecutor(
            timeout=Config.PYODIDE_TIMEOUT,
        )
        logger.info("🌐 Code execution engine: Pyodide (WASM)")

    else:
        raise ValueError(
            f"Unknown CODE_SANDBOX_ENGINE='{engine}'. "
            "Must be 'docker' or 'pyodide'."
        )

    return _executor


def execute_python_in_sandbox(code: str) -> CodeExecutionResult:
    """
    Execute a Python code string in the configured sandbox.

    This is the single public API consumed by the LangChain tool.
    """
    executor = get_executor()
    return executor.execute(code)
