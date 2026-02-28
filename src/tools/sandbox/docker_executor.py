"""
Docker-based sandbox executor.

Runs user code in a transient Docker container with heavy restrictions:
  - network_mode="none"   → no internet
  - read_only=True        → immutable root FS
  - mem_limit / cpu_count → bounded resources
  - cap_drop=["ALL"]      → minimal kernel capabilities
  - user="sandbox"        → non-root
  - tmpfs /tmp            → writable scratch space (size-limited)
"""

import logging
import os

from src.tools.sandbox.base import BaseSandboxExecutor, CodeExecutionResult

logger = logging.getLogger(__name__)


class DockerSandboxExecutor(BaseSandboxExecutor):
    """Execute Python code inside a hardened Docker container."""

    def __init__(
        self,
        image: str = "nexus-sandbox:latest",
        timeout: int = 30,
        mem_limit: str = "256m",
    ):
        self._image = image
        self._timeout = timeout
        self._mem_limit = mem_limit

    # ------------------------------------------------------------------
    # Image management
    # ------------------------------------------------------------------
    def _ensure_image(self) -> None:
        """Build the sandbox image if it doesn't exist locally."""
        import docker  # lazy import — only needed when Docker engine is used

        client = docker.from_env()

        def build_image() -> None:
            from src.core.config import Config

            dockerfile_path = os.path.join(Config.PROJECT_ROOT, "Dockerfile.sandbox")
            if not os.path.isfile(dockerfile_path):
                raise FileNotFoundError(
                    f"Dockerfile.sandbox not found at {dockerfile_path}. "
                    "Please create it or switch to CODE_SANDBOX_ENGINE=pyodide."
                )

            logger.info(f"🔨 Building sandbox image '{self._image}' …")
            client.images.build(
                path=Config.PROJECT_ROOT,
                dockerfile="Dockerfile.sandbox",
                tag=self._image,
                rm=True,
            )
            logger.info(f"✅ Sandbox image '{self._image}' built successfully.")

        # Check if image already exists
        try:
            client.images.get(self._image)
            logger.info(f"✅ Sandbox image '{self._image}' found.")
            try:
                client.containers.run(
                    image=self._image,
                    command=["python3", "-c", "import numpy, pandas, sympy, matplotlib"],
                    network_mode="none",
                    read_only=True,
                    mem_limit=self._mem_limit,
                    cpu_count=1,
                    cap_drop=["ALL"],
                    user="sandbox",
                    tmpfs={"/tmp": "size=1m,noexec,nosuid"},
                    remove=True,
                    stdout=True,
                    stderr=True,
                )
                return
            except docker.errors.ContainerError:
                logger.info(f"🔄 Sandbox image '{self._image}' is missing required packages. Rebuilding.")
                build_image()
                return
        except docker.errors.ImageNotFound:
            build_image()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute(self, code: str) -> CodeExecutionResult:
        """Run *code* inside a transient Docker container."""
        import docker
        from docker.errors import ContainerError, APIError

        self._ensure_image()

        client = docker.from_env()

        try:
            # Run the container with the code passed via the command
            # The Dockerfile's ENTRYPOINT is ["python3", "-c"]
            container = client.containers.run(
                image=self._image,
                command=["python3", "-c", code],
                # ---- Security hardening ----
                network_mode="none",
                read_only=True,
                mem_limit=self._mem_limit,
                cpu_count=1,
                cap_drop=["ALL"],
                user="sandbox",
                # Writable scratch space (1MB, exec disabled)
                tmpfs={"/tmp": "size=1m,noexec,nosuid"},
                # ---- Lifecycle ----
                detach=True,
                stdout=True,
                stderr=True,
            )

            # Wait for completion (with timeout)
            result = container.wait(timeout=self._timeout)
            exit_code = result.get("StatusCode", 1)

            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")

            return CodeExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                timed_out=False,
            )

        except Exception as exc:
            # Distinguish timeout from other errors
            exc_str = str(exc).lower()
            is_timeout = "timed out" in exc_str or "read timed out" in exc_str

            if is_timeout:
                logger.warning(f"⏱️ Docker execution timed out after {self._timeout}s.")
                return CodeExecutionResult(
                    stdout="",
                    stderr=f"Execution timed out after {self._timeout} seconds.",
                    exit_code=137,
                    timed_out=True,
                )

            logger.error(f"❌ Docker execution error: {exc}")
            return CodeExecutionResult(
                stdout="",
                stderr=f"Docker execution failed: {exc}",
                exit_code=1,
                timed_out=False,
            )

        finally:
            # Always clean up the container
            try:
                container.remove(force=True)
            except Exception:
                pass
