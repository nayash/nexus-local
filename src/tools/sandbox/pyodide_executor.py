"""
Pyodide-based sandbox executor.

Runs user code in a Pyodide (WebAssembly Python) environment via a Node.js
subprocess. Inherently sandboxed — no host FS access, no network, isolated memory.

Requirements:
  - Node.js (>= 18) installed on the host
  - npx available (comes with npm)
  - The bundled pyodide_runner.js script in this directory
"""

import json
import logging
import os
import subprocess
import shutil

from src.core.config import Config
from src.tools.sandbox.base import BaseSandboxExecutor, CodeExecutionResult

logger = logging.getLogger(__name__)

# Path to the Node.js runner script bundled alongside this module
_RUNNER_SCRIPT = os.path.join(os.path.dirname(__file__), "pyodide_runner.js")


class PyodideSandboxExecutor(BaseSandboxExecutor):
    """Execute Python code inside a Pyodide (WASM) sandbox via Node.js."""

    def __init__(self, timeout: int = 30):
        self._timeout = timeout
        self._node_bin = self._find_node()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _find_node() -> str:
        """Locate the Node.js binary."""
        node = shutil.which("node")
        if node is None:
            raise EnvironmentError(
                "Node.js is required for the Pyodide sandbox engine but was not found. "
                "Install Node.js (>= 18) or switch to CODE_SANDBOX_ENGINE=docker."
            )
        return node

    def _ensure_pyodide_installed(self) -> None:
        """
        Check if the Pyodide npm package is accessible via npx.
        On first run, npx will auto-download pyodide — we just verify Node works.
        """
        try:
            result = subprocess.run(
                [self._node_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            version = result.stdout.strip()
            logger.info(f"✅ Node.js {version} found for Pyodide engine.")
        except Exception as e:
            raise EnvironmentError(f"Node.js check failed: {e}")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute(self, code: str) -> CodeExecutionResult:
        """
        Run *code* in a Pyodide WASM sandbox.

        The code is sent to pyodide_runner.js via stdin.
        The runner outputs a JSON object with stdout, stderr, exit_code.
        """
        if not os.path.isfile(_RUNNER_SCRIPT):
            return CodeExecutionResult(
                stdout="",
                stderr=f"Pyodide runner script not found at {_RUNNER_SCRIPT}",
                exit_code=1,
                timed_out=False,
            )

        try:
            env = os.environ.copy()
            pyodide_node_modules = os.path.join(Config.PYODIDE_NPM_DIR, "node_modules")
            existing_node_path = env.get("NODE_PATH", "")
            if existing_node_path:
                env["NODE_PATH"] = f"{pyodide_node_modules}{os.pathsep}{existing_node_path}"
            else:
                env["NODE_PATH"] = pyodide_node_modules

            proc = subprocess.run(
                [self._node_bin, _RUNNER_SCRIPT],
                input=code,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=env,
            )

            # The runner script outputs JSON on stdout
            try:
                result = json.loads(proc.stdout)
                return CodeExecutionResult(
                    stdout=result.get("stdout", ""),
                    stderr=result.get("stderr", ""),
                    exit_code=result.get("exit_code", proc.returncode),
                    timed_out=False,
                )
            except json.JSONDecodeError:
                # If we can't parse JSON, return raw output
                return CodeExecutionResult(
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    exit_code=proc.returncode,
                    timed_out=False,
                )

        except subprocess.TimeoutExpired:
            logger.warning(f"⏱️ Pyodide execution timed out after {self._timeout}s.")
            return CodeExecutionResult(
                stdout="",
                stderr=f"Execution timed out after {self._timeout} seconds.",
                exit_code=137,
                timed_out=True,
            )
        except Exception as exc:
            logger.error(f"❌ Pyodide execution error: {exc}")
            return CodeExecutionResult(
                stdout="",
                stderr=f"Pyodide execution failed: {exc}",
                exit_code=1,
                timed_out=False,
            )
