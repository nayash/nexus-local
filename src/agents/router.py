from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.config import Config
from src.core.user_settings import get_setting, save_setting

# Initialize a lightweight LLM instance for routing
# (We use the same model, but with temperature=0 for strict logic)
llm = ChatOllama(
    model=get_setting("model_name", "llama3.1"),
    temperature=0, 
    base_url=Config.OLLAMA_BASE_URL
)

# Define the routing prompt
# We force it to return a single word matching our Graph Node names
route_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at routing user queries.
    Analyze the user's question and decide the next step.
    
    Options:
    - "web_search": If the user asks about current events, public news, or external technical facts (e.g., "React 19 features", "Weather").
    - "local_search": If the user asks about "Nexus-Local", "project plans", "my documents", or specific local files.
    - "generate": If the user just wants to chat, greeting, or general knowledge. E.g. If user says "Hi", "Hello", "How are you?" -- answer in a friendly manner.
    
    Output ONLY the option name. No preamble."""),
    ("human", "{question}"),
])

# Create the chain
router_chain = route_prompt | llm | StrOutputParser()

def route_question(state):
    """
    Conditional logic function for LangGraph.
    Returns the name of the next node.
    """
    print("--- 🚦 ROUTING QUERY ---")
    question = state["messages"][-1].content
    
    try:
        decision = router_chain.invoke({"question": question})
        # Clean up whitespace/newlines
        decision = decision.strip().lower()
        
        if "web_search" in decision:
            print("   -> Decision: WEB SEARCH")
            return "web_search"
        elif "local_search" in decision:
            print("   -> Decision: LOCAL SEARCH")
            return "local_search"
        else:
            print("   -> Decision: GENERATE (Skip Search)")
            return "generate"
            
    except Exception as e:
        # Fallback to search if unsure
        print(f"   -> Routing Error: {e}. Fallback to Search.")
        return "web_search"