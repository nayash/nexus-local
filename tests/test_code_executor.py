"""
Unit tests for the sandboxed code execution system.

All tests mock the underlying execution engines (Docker SDK / subprocess)
so they run without Docker or Node.js installed.
"""

import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

def _mock_config(**overrides):
    """Return a mock Config with sandbox defaults + overrides."""
    defaults = {
        "CODE_SANDBOX_ENGINE": "docker",
        "DOCKER_SANDBOX_IMAGE": "nexus-sandbox:latest",
        "DOCKER_TIMEOUT": 30,
        "DOCKER_MEM_LIMIT": "256m",
        "PYODIDE_TIMEOUT": 30,
        "PROJECT_ROOT": PROJECT_ROOT,
    }
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


# ---------------------------------------------------------------------------
# CodeExecutionResult schema tests
# ---------------------------------------------------------------------------

class TestCodeExecutionResult:
    def test_defaults(self):
        from src.tools.sandbox.base import CodeExecutionResult
        r = CodeExecutionResult()
        assert r.stdout == ""
        assert r.stderr == ""
        assert r.exit_code == 0
        assert r.timed_out is False

    def test_custom_values(self):
        from src.tools.sandbox.base import CodeExecutionResult
        r = CodeExecutionResult(stdout="hi\n", stderr="warn", exit_code=1, timed_out=True)
        assert r.stdout == "hi\n"
        assert r.stderr == "warn"
        assert r.exit_code == 1
        assert r.timed_out is True


# ---------------------------------------------------------------------------
# Factory routing tests
# ---------------------------------------------------------------------------

class TestFactory:

    def _reset_executor(self):
        """Reset the module-level singleton between tests."""
        import src.tools.code_executor as mod
        mod._executor = None

    def test_docker_engine_selected(self):
        self._reset_executor()
        with patch("src.tools.code_executor.Config", _mock_config(CODE_SANDBOX_ENGINE="docker")):
            from src.tools.code_executor import get_executor
            from src.tools.sandbox.docker_executor import DockerSandboxExecutor
            executor = get_executor()
            assert isinstance(executor, DockerSandboxExecutor)
        self._reset_executor()

    def test_pyodide_engine_selected(self):
        self._reset_executor()
        with patch("src.tools.code_executor.Config", _mock_config(CODE_SANDBOX_ENGINE="pyodide")), \
             patch("src.tools.sandbox.pyodide_executor.shutil") as mock_shutil:
            mock_shutil.which.return_value = "/usr/bin/node"
            from src.tools.code_executor import get_executor
            from src.tools.sandbox.pyodide_executor import PyodideSandboxExecutor
            executor = get_executor()
            assert isinstance(executor, PyodideSandboxExecutor)
        self._reset_executor()

    def test_unknown_engine_raises(self):
        self._reset_executor()
        with patch("src.tools.code_executor.Config", _mock_config(CODE_SANDBOX_ENGINE="unknown")):
            from src.tools.code_executor import get_executor
            with pytest.raises(ValueError, match="unknown"):
                get_executor()
        self._reset_executor()


# ---------------------------------------------------------------------------
# Docker executor tests (mocked)
# Uses `create=True` because `docker` is lazily imported inside methods,
# so the attribute does not exist at module level.
# ---------------------------------------------------------------------------

class TestDockerExecutor:

    def _make_mock_docker(self, mock_container):
        """Create a mock docker module with standard configuration."""
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_client.images.get.return_value = MagicMock()  # image exists
        
        mock_docker = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_docker.errors.ImageNotFound = type("ImageNotFound", (Exception,), {})
        return mock_docker, mock_client

    def test_happy_path(self):
        from src.tools.sandbox.docker_executor import DockerSandboxExecutor

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.side_effect = [
            b"hello world\n",  # stdout
            b"",               # stderr
        ]

        mock_docker, mock_client = self._make_mock_docker(mock_container)

        with patch.dict("sys.modules", {"docker": mock_docker, "docker.errors": mock_docker.errors}):
            executor = DockerSandboxExecutor()
            result = executor.execute('print("hello world")')

        assert result.stdout == "hello world\n"
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.timed_out is False

    def test_error_capture(self):
        from src.tools.sandbox.docker_executor import DockerSandboxExecutor

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 1}
        mock_container.logs.side_effect = [
            b"",                         # stdout
            b"SyntaxError: invalid\n",   # stderr
        ]

        mock_docker, mock_client = self._make_mock_docker(mock_container)

        with patch.dict("sys.modules", {"docker": mock_docker, "docker.errors": mock_docker.errors}):
            executor = DockerSandboxExecutor()
            result = executor.execute("invalid syntax !!!")

        assert "SyntaxError" in result.stderr
        assert result.exit_code == 1

    def test_timeout(self):
        from src.tools.sandbox.docker_executor import DockerSandboxExecutor

        mock_container = MagicMock()
        mock_container.wait.side_effect = Exception("read timed out")

        mock_docker, mock_client = self._make_mock_docker(mock_container)

        with patch.dict("sys.modules", {"docker": mock_docker, "docker.errors": mock_docker.errors}):
            executor = DockerSandboxExecutor(timeout=5)
            result = executor.execute("while True: pass")

        assert result.timed_out is True
        assert result.exit_code == 137

    def test_security_flags(self):
        """Verify container is created with the correct security flags."""
        from src.tools.sandbox.docker_executor import DockerSandboxExecutor

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.side_effect = [b"", b""]

        mock_docker, mock_client = self._make_mock_docker(mock_container)

        with patch.dict("sys.modules", {"docker": mock_docker, "docker.errors": mock_docker.errors}):
            executor = DockerSandboxExecutor()
            executor.execute("pass")

        call_kwargs = mock_client.containers.run.call_args
        assert call_kwargs.kwargs["network_mode"] == "none"
        assert call_kwargs.kwargs["read_only"] is True
        assert call_kwargs.kwargs["cap_drop"] == ["ALL"]
        assert call_kwargs.kwargs["user"] == "sandbox"
        assert call_kwargs.kwargs["mem_limit"] == "256m"


# ---------------------------------------------------------------------------
# Pyodide executor tests (mocked subprocess)
# ---------------------------------------------------------------------------

class TestPyodideExecutor:

    def _create_executor(self):
        with patch("src.tools.sandbox.pyodide_executor.shutil") as mock_shutil:
            mock_shutil.which.return_value = "/usr/bin/node"
            from src.tools.sandbox.pyodide_executor import PyodideSandboxExecutor
            return PyodideSandboxExecutor(timeout=30)

    def test_happy_path(self):
        executor = self._create_executor()
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"stdout": "42\n", "stderr": "", "exit_code": 0})
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("src.tools.sandbox.pyodide_executor.subprocess.run", return_value=mock_result), \
             patch("src.tools.sandbox.pyodide_executor.os.path.isfile", return_value=True):
            result = executor.execute("print(42)")

        assert result.stdout == "42\n"
        assert result.exit_code == 0
        assert result.timed_out is False

    def test_error_capture(self):
        executor = self._create_executor()
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"stdout": "", "stderr": "NameError: name 'x' is not defined\n", "exit_code": 1})
        mock_result.stderr = ""
        mock_result.returncode = 1

        with patch("src.tools.sandbox.pyodide_executor.subprocess.run", return_value=mock_result), \
             patch("src.tools.sandbox.pyodide_executor.os.path.isfile", return_value=True):
            result = executor.execute("print(x)")

        assert "NameError" in result.stderr
        assert result.exit_code == 1

    def test_timeout(self):
        import subprocess as real_subprocess
        executor = self._create_executor()

        with patch("src.tools.sandbox.pyodide_executor.subprocess.run", side_effect=real_subprocess.TimeoutExpired("node", 30)), \
             patch("src.tools.sandbox.pyodide_executor.os.path.isfile", return_value=True):
            result = executor.execute("while True: pass")

        assert result.timed_out is True
        assert result.exit_code == 137

    def test_missing_runner_script(self):
        executor = self._create_executor()

        with patch("src.tools.sandbox.pyodide_executor.os.path.isfile", return_value=False):
            result = executor.execute("print(1)")

        assert result.exit_code == 1
        assert "not found" in result.stderr

    def test_node_not_found(self):
        """If node is not installed, constructor should raise."""
        with patch("src.tools.sandbox.pyodide_executor.shutil") as mock_shutil:
            mock_shutil.which.return_value = None
            from src.tools.sandbox.pyodide_executor import PyodideSandboxExecutor
            with pytest.raises(EnvironmentError, match="Node.js"):
                PyodideSandboxExecutor()


# ---------------------------------------------------------------------------
# Tool registration test
# ---------------------------------------------------------------------------

class TestToolRegistration:

    def test_execute_python_code_in_tools_list(self):
        from src.tools.registry import TOOLS
        tool_names = [t.name for t in TOOLS]
        assert "execute_python_code" in tool_names

    def test_all_existing_tools_present(self):
        """Ensure we didn't break existing tool registration."""
        from src.tools.registry import TOOLS
        tool_names = [t.name for t in TOOLS]
        assert "web_search_tool" in tool_names
        assert "local_search_tool" in tool_names
        assert "get_current_time" in tool_names
