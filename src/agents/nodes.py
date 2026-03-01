import json
import re
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.messages.tool import ToolCall
from src.agents.state import AgentState
from src.agents.utils import trim_messages
from src.tools.registry import TOOLS
from langchain_ollama import ChatOllama
from src.core.config import Config
from src.core.user_settings import get_setting

# ... Initialize LLM Helper ...
def get_llm(model_name: str = "llama3.1"):
    """
    Instantiates the ChatOllama model with the requested model name.
    """
    llm = ChatOllama(
        model=model_name, # can be cloud model too, need API Key
        temperature=0,
        base_url=Config.OLLAMA_BASE_URL, # should point to https://api.ollama.com for cloud models
        # headers={"X-Thinking-Mode": "enable"} # TODO use this based on a classifier
    )
    return llm.bind_tools(TOOLS)

# SYSTEM_PROMPT = """You are Nexus, a smart research assistant.

# RULES:
# 1. **Tool Reliance:** If a tool returns information (like the date), it is the ABSOLUTE TRUTH. It overrides everything you know.
# 2. **Step-by-Step Math:** When asked about age or time, you MUST show your work: "Current Year - Birth Year = Age".
# 3. **No Cutoff Excuses:** Never mention your "knowledge cutoff". Just solve the problem.
# 4. **Direct Action:** DO NOT narrate what you are going to do. Just call the tool.
# 5. **Don't Use Stale Training Data:** For any query which requires current information, use the tools to get the information.
# """

SYSTEM_PROMPT = """You are Nexus, a specialized research assistant with access to the user's local private files and the public web.

### CRITICAL PROTOCOLS (MUST FOLLOW):

1. **CONVERSATIONAL CONTEXT & COREFERENCE RESOLUTION:**
   - You have access to the FULL conversation history. Read it carefully before responding.
   - When the user says "this", "that", "it", "the paper", "the document" etc., look at your PREVIOUS responses to identify what they're referring to.
   - If your previous response mentioned a specific local file, and the user asks a follow-up question about "it" or "this", use `local_search_tool` with that file's name in the query.
   - Example:
     * User: "What is the abstract of GAN paper.pdf?"
     * You: [Call local_search_tool, return abstract]
     * User: "Summarize this paper"
     * You: [Understand "this paper" = "GAN paper.pdf", call local_search_tool with query "summarize GAN paper.pdf"]

2. **LOCAL-FIRST STRATEGY:**
   - Your unique advantage is access to private files.
   - If a query mentions a specific filename or references a previously discussed file, you MUST use `local_search_tool`.
   - Only use `web_search_tool` if the local search returns no results or if the user explicitly asks for public info.

3. **HYBRID SEARCH:**
   - If a query could be both public and private, call BOTH tools to provide comprehensive answers.
   - DO NOT tell user to search a website. Access the website yourself and provide the answer.

4. **TOOL TRUTH:**
   - Information returned by tools (Dates, Content) is the ABSOLUTE TRUTH.
   - It overrides your internal training data. Never mention "knowledge cutoff".

5. **MANDATORY MATH:**
   - For age/time questions, call `get_current_time` first.
   - Show your work: "Current Year (from tool) - Birth Year = Age".

6. **DIRECT EXECUTION:**
   - DO NOT narrate your plan (e.g., "I will now search...").
   - Just call the tool immediately.

7. **CASUAL CONVERSATION:**
   - For greetings or "getting to know you" questions, DO NOT USE TOOLS.
   - For questions about your identity, search data/nexus-identity.txt in lanceDB.
   - DON'T OUTPUT your thoughts to the user.

8. **ERROR HANDLING & FORMATTING:**
   - Try ONE alternative or apologize. Infinite retries are forbidden.
   - **CRITICAL**: DO NOT output raw JSON strings (e.g. `{"name": ...}`). ALWAYS use the native tool calling capability.
   - If you see a tool call in the output but it's just text, you failed. Use the proper tool structure.
   - For short factual questions (e.g. "who is protagonist?", "what is the title?"), answer directly in the first sentence.
   - Do NOT start with filler like "The provided text..." or give a broad summary unless the user asked for one.

9. **SOURCES:**
   - DO NOT list sources in your response.
   - The system handles source citations automatically.

10. **STRICT FILE FILTER RULES:**
    - `file_filter` in `local_search_tool` is ONLY for explicitly attached/focused files.
    - If user mentions a filename buts hasn't attached it, include the filename in the `query` argument, NOT `file_filter`.
    - Leave `file_filter` as None unless the user has attached a file to the chat.

11. **QUERY PRESERVATION & INTENT:**
    - The search engine uses natural language understanding (NLU) for metadata filtering.
    - You MUST pass the FULL user question including the INTENT (e.g., "Summarize", "Find author", "List key points").
    - **INCORRECT**: `local_search_tool(query="GAN paper.pdf")` -> LOSES "Summarize" intent!
    - **CORRECT**: `local_search_tool(query="Summarize the GAN paper.pdf")` -> PRESERVES intent.
    - **INCORRECT**: `local_search_tool(query="Project report")`
    - **CORRECT**: `local_search_tool(query="Find the author of the Project report")`

12. **CRITICAL: STAY FOCUSED ON USER'S ACTUAL QUERY:**
    - Tool results may contain irrelevant text from logs or past conversations.
    - ALWAYS respond to the USER'S ORIGINAL QUESTION, not random content you see in tool results.
    - If tool results contain text about unrelated topics (e.g., user asks about "GAN.pdf" but results mention "puzzle games"), IGNORE the irrelevant content.
    - If the requested file is not found, acknowledge it clearly: "The file [filename] was not found in local storage."
    - NEVER hallucinate that the user asked about topics you only saw in tool result logs.

13. **CODE EXECUTION:**
    - If a query requires computation, data processing, math, or logic that cannot be answered by web/local search, use `execute_python_code`.
    - Write clean, complete, self-contained Python scripts. Print the final answer to stdout using `print()`.
    - The code runs in a highly restricted sandbox (no network access, no host file access).
    - Assume standard data science libraries are available: `numpy`, `pandas`, `sympy`.
    - DO NOT use this for questions that can be answered directly, via search, or from your training data.
    - If the code execution fails, try ONE alternative approach or explain the issue to the user.
"""


# Module-level cache for LLM instances to avoid frequent re-initialization
_llm_cache = {}


def _get_last_user_message_content(messages) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human":
            return getattr(msg, "content", "") or ""
    return ""


def _get_latest_message_type(messages) -> str:
    if not messages:
        return ""
    return getattr(messages[-1], "type", "") or ""


def _collect_recent_tool_contents(messages, limit: int = 3) -> list[str]:
    contents = []
    for msg in reversed(messages):
        if getattr(msg, "type", None) != "tool":
            continue
        contents.append(getattr(msg, "content", "") or "")
        if len(contents) >= limit:
            break
    return contents


def _is_short_factual_question(text: str) -> bool:
    normalized = (text or "").strip()
    if not normalized:
        return False

    compact = " ".join(normalized.lower().split())
    word_count = len(compact.split())
    if len(compact) > 140 or word_count > 16:
        return False

    factual_prefixes = (
        "who ",
        "what ",
        "when ",
        "where ",
        "which ",
        "whose ",
        "is ",
        "are ",
        "did ",
        "does ",
        "do ",
        "how did ",
        "how does ",
        "how many ",
        "how much ",
    )
    return compact.endswith("?") or compact.startswith(factual_prefixes)


def _strip_reasoning_artifacts(content: str) -> str:
    cleaned = (content or "").strip()
    if not cleaned:
        return cleaned

    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()

    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"^[^\w\"'(*\[]+\s*", "", cleaned)
    return cleaned.strip()


def _should_use_reasoning_mode(messages) -> bool:
    last_user_content = _get_last_user_message_content(messages)
    latest_type = _get_latest_message_type(messages)
    normalized = " ".join((last_user_content or "").lower().split())
    words = normalized.split()
    word_count = len(words)
    char_count = len(normalized)
    question_count = last_user_content.count("?")

    if _is_short_factual_question(last_user_content):
        return False

    strong_complex_patterns = (
        "summarize",
        "summary",
        "analyze",
        "analysis",
        "explain",
        "interpret",
        "comparison",
        "compare",
        "contrast",
        "critique",
        "evaluate",
        "reason about",
        "walk me through",
        "step by step",
        "pros and cons",
        "key themes",
        "main arguments",
    )
    if any(pattern in normalized for pattern in strong_complex_patterns):
        return True

    if question_count > 1:
        return True

    if char_count > 220 or word_count > 32:
        return True

    conjunction_count = sum(normalized.count(token) for token in (" and ", " or ", " then ", " also "))
    if conjunction_count >= 2:
        return True

    if normalized.startswith("why "):
        return True

    if normalized.startswith("how "):
        simple_how_patterns = (
            "how many",
            "how much",
            "how old",
            "how long",
            "how far",
            "how tall",
            "how big",
            "how did the story end",
        )
        if not any(normalized.startswith(pattern) for pattern in simple_how_patterns):
            return True

    if latest_type == "tool":
        tool_contents = _collect_recent_tool_contents(messages, limit=3)
        combined_tool_size = sum(len(content) for content in tool_contents)
        if combined_tool_size > 3500:
            return True
        if len(tool_contents) > 1:
            return True
        if tool_contents and any(marker in tool_contents[0].lower() for marker in (
            "summary",
            "analysis",
            "key points",
            "content truncated",
        )):
            return True

    return False


def get_cached_llm(model_name: str, with_tools: bool = True, thinking_mode: bool = False):
    cache_key = f"{model_name}_{with_tools}_{thinking_mode}"
    if cache_key not in _llm_cache:
        print(f"   ⚙️ Initializing LLM: {model_name} (Tools: {with_tools}, Thinking: {thinking_mode})")
        init_kwargs = {
            "model": model_name,
            "temperature": 0,
            "base_url": Config.OLLAMA_BASE_URL,
        }
        if thinking_mode:
            init_kwargs["headers"] = {"X-Thinking-Mode": "enable"}
        llm = ChatOllama(**init_kwargs)
        if with_tools:
            _llm_cache[cache_key] = llm.bind_tools(TOOLS)
        else:
            _llm_cache[cache_key] = llm
            
    return _llm_cache[cache_key]

def agent_node(state: AgentState):
    print("--- 🤖 NODE: AGENT ---")
    
    # Trim messages to keep context window manageable for local LLM
    trimmed_history = trim_messages(state["messages"], max_messages=12)
    
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + trimmed_history
    print(f'messages: size: {len(messages)} --> {messages}')
    last_user_content = _get_last_user_message_content(messages)
    latest_message_type = _get_latest_message_type(messages)
    use_reasoning_mode = latest_message_type == "tool" and _should_use_reasoning_mode(messages)
    print(f"agent_node: latest_message_type={latest_message_type}, reasoning_mode={use_reasoning_mode}")
    # 1. Retrieve LLM based on UserSettings
    current_model = get_setting("model_name", "llama3.1")
    llm_instance = get_cached_llm(current_model, with_tools=True, thinking_mode=use_reasoning_mode)
        
    # --- FOCUS MODE CHECK ---
    # NOW: Read from state, not session
    print(f'agent_node: focused_file: {state.get("focused_file")}')
    focused_file = state.get("focused_file")
    
    if focused_file:
        print(f"   🎯 FOCUS MODE ACTIVE: {focused_file}")
        is_tabular_focus = focused_file.lower().endswith((".csv", ".tsv", ".xlsx", ".xls"))
        focus_instruction = (
            f"\n\n### 🚨 FOCUS MODE ACTIVE 🚨\n"
            f"The user is strictly focusing on the file: '{focused_file}'.\n"
            + (
                f"You MUST use `analyze_tabular_file_tool` with `file_path='{focused_file}'` for queries about this file.\n"
                f"Pass the user's full request as `user_query`.\n"
                f"Do NOT use `local_search_tool` or `execute_python_code` for this tabular file.\n"
                if is_tabular_focus
                else f"You MUST strictly use the `local_search_tool` with `file_filter='{focused_file}'` for every query.\n"
            )
            + (
                f"If the tool returns no information (or explicitly says so), you MUST state: 'The attached document does not contain this information.'\n"
                f"DO NOT use `web_search_tool` or general knowledge while in this mode."
            )
        )
        # Modify the System Message (first message)
        if isinstance(messages[0], SystemMessage):
            # Create a new system message with the added instruction
            new_content = messages[0].content + focus_instruction
            messages[0] = SystemMessage(content=new_content)

    if _is_short_factual_question(last_user_content) and isinstance(messages[0], SystemMessage):
        concise_instruction = (
            "\n\n### CONCISE FACT MODE\n"
            "The user's latest message is a short factual question.\n"
            "After using any needed tool results, answer in 1 sentence first.\n"
            "Keep it under 40 words unless the user explicitly asks for detail.\n"
            "Do not provide a summary, analysis, or extra background unless requested.\n"
        )
        messages[0] = SystemMessage(content=messages[0].content + concise_instruction)
    
    # 2. Invoke the LLM
    response = llm_instance.invoke(messages)
    print(f'llm response: {response}')

    # --- 🛡️ QUERY ENRICHMENT LOGIC ---
    # Fix LLM truncating query to just filenames (e.g., query="GAN.pdf" vs. "Summarize GAN.pdf")
    # This ensures the search engine gets the FULL intent for metadata filtering & semantic search.
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "local_search_tool":
                args = tool_call.get("args", {})
                query = args.get("query", "")

                # Heuristic: 
                # 1. Query looks like a filename (ends in extension)
                # 2. OR Query is significantly shorter than user message (loss of context)
                # 3. AND User message is not massive (>1000 chars)
                
                is_filename_only = re.match(r'^[\w\-. ]+\.(pdf|txt|md|csv|sh)$', query.strip(), re.IGNORECASE)
                is_truncated = len(query) < len(last_user_content) * 0.5
                
                if (is_filename_only or is_truncated) and 0 < len(last_user_content) < 1000:
                    print(f"   ✨ ENRICHING QUERY: '{query}' -> '{last_user_content}'")
                    # Update the args in place
                    # Note: tool_call is a dict-like object in recent LangChain versions or a ToolCall dict
                    if isinstance(tool_call, dict):
                         tool_call["args"]["query"] = last_user_content
                    else:
                         # If it's an object (depending on version), might need different handling
                         # But typically response.tool_calls is a list of ToolCall dicts
                         tool_call["args"]["query"] = last_user_content

        if focused_file and focused_file.lower().endswith((".csv", ".tsv", ".xlsx", ".xls")):
            for tool_call in response.tool_calls:
                if tool_call["name"] in {"local_search_tool", "execute_python_code"}:
                    print(f"   🔀 REWRITING TOOL CALL: {tool_call['name']} -> analyze_tabular_file_tool")
                    replacement_args = {
                        "file_path": focused_file,
                        "user_query": last_user_content,
                    }
                    if isinstance(tool_call, dict):
                        tool_call["name"] = "analyze_tabular_file_tool"
                        tool_call["args"] = replacement_args
                    else:
                        tool_call["name"] = "analyze_tabular_file_tool"
                        tool_call["args"] = replacement_args

    # --- 🛡️ RESCUE LOGIC: Fix "Chatty" Tool Calls ---
    # If the model didn't trigger a native tool call, check if it wrote JSON in the text.
    if not response.tool_calls and response.content:
        content = response.content.strip()
        
        # Simple heuristic: starts with { and contains "name" and ("parameters" or "args")
        if content.startswith("{") and '"name":' in content and ('"parameters":' in content or '"args":' in content):
             print("   ⚠️ DETECTED CHATTY TOOL CALL. RESCUING...")
             try:
                # Find the JSON object
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                json_str = content[json_start:json_end]
                
                # Fix Python None -> null for valid JSON
                json_str_fixed = json_str.replace(": None", ": null").replace(": True", ": true").replace(": False", ": false")
                
                tool_data = json.loads(json_str_fixed)
                
                args = tool_data.get("parameters") or tool_data.get("args", {})
                
                tool_call = ToolCall(
                    name=tool_data["name"],
                    args=args,
                    id="call_rescue_" + str(hash(json_str))
                )
                
                response.tool_calls = [tool_call]
                response.content = "" # Silence the chatty output
                
             except Exception as e:
                print(f"   ❌ RESCUE FAILED: {e}")

    if not response.tool_calls and response.content:
        cleaned_content = _strip_reasoning_artifacts(response.content)
        if cleaned_content != response.content:
            response.content = cleaned_content

    return {"messages": [response]}
