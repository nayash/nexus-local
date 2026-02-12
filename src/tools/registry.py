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
        description="The type of search. MUST be one of: 'general', 'news', 'science', 'it'. Default to 'general' if unsure."
    )
    time_range: Literal["", "day", "week", "month", "year"] = Field(
        default="",
        description="Filter by time. MUST be one of: '' (anytime), 'day', 'week', 'month', 'year'. Do NOT use 'now' or 'recent'."
    )

class LocalSearchInput(BaseModel):
    query: str = Field(description="The full search query. Do NOT shorten the query to keywords. Pass the full user question or a comprehensive search query.")
    file_filter: Optional[str] = Field(
        default=None,
        description="The absolute path of a specific file to search within. MUST be a valid, existing path provided in the context (e.g. from `focused_file` state). DO NOT guess or hallucinate paths based on filenames in the query. If the user mentions a filename but has not attached it, include the filename in the `query` field and leave this `file_filter` as None."
    )

@tool(args_schema=SearchInput, response_format="content_and_artifact")
def web_search_tool(query: str, category: str = "general", time_range: str = ""):
    """
    Search the web with filters. 
    """
    # 1. Call the function that returns Objects
    results_objects = search_web(query, category, time_range)
    
    # 2. Convert Objects to String for the LLM
    if not results_objects:
        return "No results found. DO NOT TRY AGAIN with the same query. Try a significantly different query or answer based on your knowledge if allowed."
        
    formatted_string = ""
    # Metadata for sources
    source_metadata = []
    
    for res in results_objects:
        # Assuming your SearchResult model has these fields
        formatted_string += f"Title: {res.title}\nURL: {res.url}\nContent: {res.content}\n\n"
        source_metadata.append({"title": res.title, "url": res.url, "type": "web"})
    
    filtered_content = mini_rag_filter(formatted_string, query)
    
    # Return tuple: (Content, Metadata List)
    # With response_format="content_and_artifact", LangChain treats the first element
    # as the content for the LLM, and the second as the artifact for the system.
    return filtered_content, source_metadata

@tool(args_schema=LocalSearchInput, response_format="content_and_artifact")
def local_search_tool(query: str, file_filter: str = None):
    """
    Search the user's local private documents and files.
    Use this for questions about "Nexus", "Project", or personal data.
    """
    print(f'calling search_local with query: {query} and file_filter: {file_filter}')
    results = search_local(query, file_filter)
    
    # Increase context budget for focused search to 15,000 chars
    context_budget = 15000 if file_filter else 10000
    
    # Extract metadata first
    source_metadata = []
    for r in results:
        source_metadata.append({
            "title": r.title, 
            "url": r.url, # Check if searches return absolute path in url field
            "type": "local"
        })
        
    context_str = "\n\n".join([r.to_context_string() for r in results])
    
    # OPTIMIZATION: For local search (especially parent strategy), we bypass mini_rag_filter.
    # Paragraph-level keyword filtering often breaks the cohesive context retrieved by the 
    # parent strategy. We use simple truncation as a safety measure instead.
    if len(context_str) > context_budget:
        context_str = context_str[:context_budget] + "\n\n... [Content truncated to fit context window] ..."
        
    return context_str, source_metadata

@tool
def get_current_time():
    """
    Get the current date and time.
    Use this for all relative time calculations.
    """
    now = datetime.now()
    # We return a string that explicitly tells the LLM what to do
    return f"The Current Date and Time is {now.strftime('%Y-%m-%d %H:%M:%S')}. You MUST use the year {now.year} for all age calculations. Ignore your training data."

# List of tools available to the brain
TOOLS = [web_search_tool, local_search_tool, get_current_time]