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
        model=model_name,
        temperature=0,
        base_url=Config.OLLAMA_BASE_URL,
        headers={"X-Thinking-Mode": "enable"}
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

1. **LOCAL-FIRST STRATEGY:** - Your unique advantage is access to private files.
   - If a query contains specific terms (e.g., "NestedLearning", "Project Alpha"), you MUST check `local_search_tool` FIRST.
   - Only use `web_search_tool` if the local search returns no results or if the user explicitly asks for public info.

2. **HYBRID SEARCH:** If a query seems like it could be both public and private (e.g., "React patterns"), you are allowed to call BOTH `local_search_tool` AND `web_search_tool` to provide a comprehensive answer.
   - DO NOT tell user to search a website. If you know where to find the data, access the website yourself and provide answer.
   
3. **ReAct:** You are allowed to call tools based on your own reasoning. Your absolute priority is to solve user's query.
   - For instance, if user asks "Give me top 5 global News for today", you can first call web search with category and time_range as "news" and "day".
   Then again, access the URLs of results using web_search_tool and provide the 5 top news items.

4. **TOOL TRUTH:**
   - Information returned by tools (Dates, Content) is the ABSOLUTE TRUTH.
   - It overrides your internal training data. Never mention "knowledge cutoff".

5. **MANDATORY MATH:** - For age/time or schedule questions, you MUST call `get_current_time` first to know current date time.
   - Then, show your work: "Current Year (from tool) - Birth Year = Age".

6. **DIRECT EXECUTION:**
   - DO NOT narrate your plan (e.g., "I will now search...").
   - Just output the tool call immediately.
   
7. **NO LOOPS:**
   - If a tool returns "No results" or an error, do NOT call the same tool again with the same arguments.
   - Try ONE alternative or apologize to the user. Infinite retries are forbidden.
"""


# Module-level cache for LLM instances to avoid frequent re-initialization
_llm_cache = {}

def get_cached_llm(model_name: str, with_tools: bool = True):
    cache_key = f"{model_name}_{with_tools}"
    if cache_key not in _llm_cache:
        print(f"   ⚙️ Initializing LLM: {model_name} (Tools: {with_tools})")
        llm = ChatOllama(
            model=model_name,
            temperature=0,
            base_url=Config.OLLAMA_BASE_URL,
            headers={"X-Thinking-Mode": "enable"}
        )
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
    # 1. Retrieve LLM based on UserSettings
    current_model = get_setting("model_name", "llama3.1")
    current_model = get_setting("model_name", "llama3.1")
    llm_instance = get_cached_llm(current_model, with_tools=True)
        
    # --- FOCUS MODE CHECK ---
    # NOW: Read from state, not session
    focused_file = state.get("focused_file")
    
    if focused_file:
        print(f"   🎯 FOCUS MODE ACTIVE: {focused_file}")
        focus_instruction = (
            f"\n\n### 🚨 FOCUS MODE ACTIVE 🚨\n"
            f"The user is strictly focusing on the file: '{focused_file}'.\n"
            f"You MUST strictly use the `local_search_tool` with `file_filter='{focused_file}'` for every query.\n"
            f"If the tool returns no information (or explicitly says so), you MUST state: 'The attached document does not contain this information.'\n"
            f"DO NOT use `web_search_tool` or general knowledge while in this mode."
        )
        # Modify the System Message (first message)
        if isinstance(messages[0], SystemMessage):
            # Create a new system message with the added instruction
            new_content = messages[0].content + focus_instruction
            messages[0] = SystemMessage(content=new_content)
    
    # 2. Invoke the LLM
    response = llm_instance.invoke(messages)
    print(f'llm response: {response}')
    # --- 🛡️ RESCUE LOGIC: Fix "Chatty" Tool Calls ---
    # If the model didn't trigger a native tool call, check if it wrote JSON in the text.
    if not response.tool_calls and response.content:
        content = response.content
        
        # Regex to find a JSON-like block: {"name": "...", "parameters": {...}}
        # We look for the pattern {"name": "..." 
        json_pattern = r'\{"name":\s*"[^"]+",\s*"parameters":\s*\{.*?\}\}'
        match = re.search(json_pattern, content, re.DOTALL)
        
        if match:
            print("   ⚠️ DETECTED CHATTY TOOL CALL. RESCUING...")
            json_str = match.group(0)
            try:
                tool_data = json.loads(json_str)
                
                # Construct a valid ToolCall object manually
                tool_call = ToolCall(
                    name=tool_data["name"],
                    args=tool_data.get("parameters", {}),
                    id="call_rescue_" + str(hash(json_str)) # Unique ID
                )
                
                # Attach it to the response so the Graph sees it
                response.tool_calls = [tool_call]
                
                # IMPORTANT: Clear the content so we don't stream the "To answer..." text to the user
                # OR keep it if you want the thought visible. 
                # Ideally, clear it so the tool runs cleanly.
                response.content = "" 
                
            except Exception as e:
                print(f"   ❌ RESCUE FAILED: {e}")

    return {"messages": [response]}