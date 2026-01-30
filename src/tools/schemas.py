from typing import Optional
from pydantic import BaseModel, HttpUrl

class SearchResult(BaseModel):
    """
    Strict definition of what a Search Result must look like.
    """
    title: str
    url: str  # Pydantic will validate this is a real URL
    content: Optional[str] = None # Some results might just be titles
    source: str = "web"

    def to_context_string(self) -> str:
        """Helper to format this result for the LLM prompt."""
        return f"Source: {self.title}\nURL: {self.url}\nContent: {self.content}\n---"