from langchain_core.tools import tool
from datetime import datetime
from src.tools.search import search_web
from src.tools.local import search_local
from pydantic import BaseModel, Field
from typing import Literal, Optional
from src.agents.utils import mini_rag_filter

class SearchInput(BaseModel):
    query: str = Field(description="The specific search query.")
    category: Literal["general", "news", "science", "it"] = Field(
        default="general", 
        description="The type of search. Use 'news' for recent events, 'science' for papers, 'it' for coding/tech, 'general' for facts."
    )
    time_range: Literal["", "day", "week", "month", "year"] = Field(
        default="",
        description="Filter results by time. Use 'day' for breaking news, 'year' for recent history."
    )

class LocalSearchInput(BaseModel):
    query: str = Field(description="The specific search query.")
    file_filter: Optional[str] = Field(
        default=None,
        description="The absolute path of a specific file to search within. Use this ONLY when the user is focusing on a specific document."
    )

@tool(args_schema=SearchInput)
def web_search_tool(query: str, category: str = "general", time_range: str = ""):
    """
    Search the web with filters. 
    """
    # 1. Call the function that returns Objects
    results_objects = search_web(query, category, time_range)
    
    # 2. Convert Objects to String for the LLM
    if not results_objects:
        return "No results found."
        
    formatted_string = ""
    for res in results_objects:
        # Assuming your SearchResult model has these fields
        formatted_string += f"Title: {res.title}\nURL: {res.url}\nContent: {res.content}\n\n"
        
    return mini_rag_filter(formatted_string, query)

@tool(args_schema=LocalSearchInput)
def local_search_tool(query: str, file_filter: str = None):
    """
    Search the user's local private documents and files.
    Use this for questions about "Nexus", "Project", or personal data.
    """
    results = search_local(query, file_filter)
    # Increase context budget for focused search to 15,000 chars
    context_budget = 15000 if file_filter else 5000
    return mini_rag_filter("\n\n".join([r.to_context_string() for r in results]), query, max_chars=context_budget)

@tool
def get_current_time():
    """
    Get the current date and time.
    Use this for all relative time calculations.
    """
    now = datetime.now()
    # We return a string that explicitly tells the LLM what to do
    return f"The Current Date is {now.strftime('%Y-%m-%d')}. You MUST use the year {now.year} for all age calculations. Ignore your training data."

# List of tools available to the brain
TOOLS = [web_search_tool, local_search_tool, get_current_time]