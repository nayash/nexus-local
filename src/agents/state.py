from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    The memory of the agent.
    
    Attributes:
        messages: A list of messages (User, AI, System, Tool). 
                  'add' reducer means new messages are appended, not overwritten.
        context:  Raw string data retrieved from search/local DB.
    """
    # operator.add ensures that when a node returns a message, it is APPENDED to the history
    messages: Annotated[List[BaseMessage], add] 
    context: str
    sources: Annotated[List[Dict[str, Any]], add]
    focused_file: str = None  # Optional: File path for focus mode