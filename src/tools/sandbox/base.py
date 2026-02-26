"""
Abstract base for sandbox executors and shared result schema.
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


class CodeExecutionResult(BaseModel):
    """Structured result from executing Python code in a sandbox."""
    stdout: str = Field(default="", description="Standard output captured from the code.")
    stderr: str = Field(default="", description="Standard error captured from the code.")
    exit_code: int = Field(default=0, description="Process exit code (0 = success).")
    timed_out: bool = Field(default=False, description="Whether execution was killed due to timeout.")


class BaseSandboxExecutor(ABC):
    """
    Strategy interface for code execution sandboxes.
    
    Each concrete executor MUST guarantee:
      - No network access
      - No access to host filesystem
      - Memory and CPU limits
      - Execution timeout
    """

    @abstractmethod
    def execute(self, code: str) -> CodeExecutionResult:
        """
        Execute a Python code string in the sandbox.

        Args:
            code: The Python source code to execute.

        Returns:
            CodeExecutionResult with captured stdout, stderr, exit code, and timeout flag.
        """
        ...
