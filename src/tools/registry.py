from langchain_core.tools import tool
from datetime import datetime
from src.tools.search import search_web
from src.tools.local import search_local
from src.tools.tabular import (
    load_tabular_dataframe,
    generate_tabular_analysis_code,
    execute_tabular_analysis,
    extract_tabular_result_payload,
    format_tabular_result_content,
    RESULT_MARKER,
)
from pydantic import BaseModel, Field
from typing import Literal
from src.agents.utils import mini_rag_filter
import os

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
    query: str = Field(description="The FULL natural language query. Ensure you include the action/intent (e.g., 'Summarize...', 'Find author of...'). DO NOT shorten the query to just keywords or filenames. The underlying search engine uses semantic search and metadata filtering, so natural language is required.")
    file_filter: str = Field(
        default="",
        description="The absolute path of a specific file to search within. MUST be a valid, existing path provided in the context (e.g. from `focused_file` state). DO NOT guess or hallucinate paths based on filenames in the query. If the user mentions a filename but has not attached it, include the filename in the `query` field and leave this `file_filter` as an empty string."
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
def local_search_tool(query: str, file_filter: str = ""):
    """
    Search the user's local private documents and files.
    Use this for questions about "Nexus", "Project", or personal data.
    """
    normalized_file_filter = (file_filter or "").strip() or None
    print(f'calling search_local with query: {query} and file_filter: {normalized_file_filter}')
    results = search_local(query, normalized_file_filter)
    
    # Increase context budget for focused search to 15,000 chars
    context_budget = 15000 if normalized_file_filter else 10000
    
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
    
    # Framing header: explicitly tell the LLM that this content was retrieved from
    # the user's indexed files — NOT pasted by the user. This prevents the LLM from
    # saying "The text you've provided is from..." when answering questions like
    # "Do I have any books by X?" where the user never shared any text themselves.
    framing_header = (
        "The following content was automatically retrieved from the user's locally indexed files "
        "in response to their query. The user did NOT paste or share this text directly.\n"
        "Answer ONLY the user's original question using this retrieved content as your source.\n"
        "If the user asks a short factual question (for example who/what/when/where/which), "
        "give the direct answer in the first sentence.\n"
        "Do NOT start with phrases like 'The provided text' and do NOT summarize the document "
        "unless the user explicitly asks for a summary or analysis.\n"
        "─────────────────────────────────────────\n"
    )
    context_str = framing_header + context_str

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


# --- Code Execution Tool ---
class CodeExecutionInput(BaseModel):
    code: str = Field(
        description=(
            "Complete, self-contained Python code to execute. "
            "The code MUST print its final answer to stdout via print(). "
            "Available libraries: numpy, pandas, sympy. No network access."
        )
    )

@tool(args_schema=CodeExecutionInput)
def execute_python_code(code: str) -> str:
    """
    Execute Python code in a secure, isolated sandbox and return the output.
    Use this for computation, math, data analysis, or any logic that
    cannot be answered by web/local search alone.
    The sandbox has NO network access and NO access to user files.
    Always print() your final answer.
    """
    from src.tools.code_executor import execute_python_in_sandbox
    result = execute_python_in_sandbox(code)
    
    if result.timed_out:
        return f"⏱️ Code execution timed out after the allowed time limit.\nPartial stderr:\n{result.stderr}"
    
    output_parts = []
    if result.stdout:
        output_parts.append(f"Output:\n{result.stdout}")
    if result.stderr:
        output_parts.append(f"Errors:\n{result.stderr}")
    if result.exit_code != 0 and not output_parts:
        output_parts.append(f"Code exited with error code {result.exit_code}.")
    
    return "\n".join(output_parts) if output_parts else "Code executed successfully with no output."


class TabularAnalysisInput(BaseModel):
    file_path: str = Field(
        description=(
            "Absolute path to a local tabular file explicitly attached or focused by the user. "
            "Supported formats: .csv, .tsv, .xlsx, .xls."
        )
    )
    user_query: str = Field(
        description=(
            "The user's full natural-language request about the file. Preserve the full intent."
        )
    )


@tool(args_schema=TabularAnalysisInput, response_format="content_and_artifact")
def analyze_tabular_file_tool(file_path: str, user_query: str):
    """
    Load a local tabular file on the host, reconstruct it as `df` inside the
    sandbox, and run LLM-generated pandas analysis code against it.
    """
    try:
        df, abs_path = load_tabular_dataframe(file_path)
    except Exception as exc:
        return (
            str(exc),
            [{"title": "Tabular Analysis Error", "url": file_path, "type": "local"}],
        )

    try:
        generated_code = generate_tabular_analysis_code(user_query, df)
    except Exception as exc:
        return (
            f"Failed to generate analysis code for '{abs_path}': {exc}",
            [{"title": os.path.basename(abs_path), "url": abs_path, "type": "local"}],
        )

    try:
        result = execute_tabular_analysis(df, generated_code)
    except Exception as exc:
        return (
            f"Failed to run sandbox analysis for '{abs_path}': {exc}",
            [{"title": os.path.basename(abs_path), "url": abs_path, "type": "local"}],
        )

    payload = None
    raw_stdout = result.stdout or ""
    try:
        payload = extract_tabular_result_payload(raw_stdout)
        if payload:
            raw_stdout = "\n".join(
                line for line in raw_stdout.splitlines() if not line.startswith(RESULT_MARKER)
            ).strip()
    except Exception as exc:
        raw_stdout = f"{raw_stdout}\nPayload parse error: {exc}".strip()

    content = format_tabular_result_content(
        user_query=user_query,
        abs_path=abs_path,
        df=df,
        generated_code=generated_code,
        payload=payload,
        raw_stdout=raw_stdout,
        stderr=result.stderr or "",
        timed_out=result.timed_out,
        exit_code=result.exit_code,
    )

    source_metadata = [{
        "title": os.path.basename(abs_path),
        "url": abs_path,
        "type": "local",
    }]
    if payload and payload.get("kind") == "plot" and payload.get("image_base64"):
        source_metadata.append({
            "type": "plot",
            "mime": "image/png",
            "image_base64": payload["image_base64"],
            "summary": payload.get("summary", ""),
            "title": f"Plot for {os.path.basename(abs_path)}",
        })
    return content, source_metadata


# List of tools available to the brain
TOOLS = [web_search_tool, local_search_tool, get_current_time, analyze_tabular_file_tool, execute_python_code]
